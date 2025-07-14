"""
Real-time Data Processing Service
Handles WebSocket connections, streaming data, and live updates
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
import logging

from app.services.file_service import FileService
from app.services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept a WebSocket connection"""
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid.uuid4())
        
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.now(),
            "subscriptions": set(),
            "last_activity": datetime.now()
        }
        
        logger.info(f"Client {client_id} connected")
        return client_id
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
                self.connection_metadata[client_id]["last_activity"] = datetime.now()
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any], subscription: str = None):
        """Broadcast a message to all connected clients or subscribers"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            # Check if client is subscribed to this type of message
            if subscription and subscription not in self.connection_metadata[client_id]["subscriptions"]:
                continue
            
            try:
                await websocket.send_text(json.dumps(message))
                self.connection_metadata[client_id]["last_activity"] = datetime.now()
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe(self, client_id: str, subscription: str):
        """Subscribe a client to a specific type of message"""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]["subscriptions"].add(subscription)
    
    def unsubscribe(self, client_id: str, subscription: str):
        """Unsubscribe a client from a specific type of message"""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]["subscriptions"].discard(subscription)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)

class RealTimeService:
    """Service for real-time data processing and streaming"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.file_service = FileService()
        self.analysis_service = AnalysisService()
        self.streaming_datasets: Dict[str, Dict[str, Any]] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    async def handle_websocket_connection(self, websocket: WebSocket, client_id: str = None):
        """Handle a new WebSocket connection"""
        client_id = await self.connection_manager.connect(websocket, client_id)
        
        try:
            # Send welcome message
            await self.connection_manager.send_personal_message({
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to AssureTheAnalyst real-time service"
            }, client_id)
            
            # Listen for messages
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await self.handle_message(client_id, message)
                
        except WebSocketDisconnect:
            self.connection_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
            self.connection_manager.disconnect(client_id)
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            subscription = message.get("subscription")
            if subscription:
                self.connection_manager.subscribe(client_id, subscription)
                await self.connection_manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "subscription": subscription,
                    "timestamp": datetime.now().isoformat()
                }, client_id)
        
        elif message_type == "unsubscribe":
            subscription = message.get("subscription")
            if subscription:
                self.connection_manager.unsubscribe(client_id, subscription)
                await self.connection_manager.send_personal_message({
                    "type": "unsubscription_confirmed",
                    "subscription": subscription,
                    "timestamp": datetime.now().isoformat()
                }, client_id)
        
        elif message_type == "start_monitoring":
            dataset_id = message.get("dataset_id")
            if dataset_id:
                await self.start_dataset_monitoring(dataset_id, client_id)
        
        elif message_type == "stop_monitoring":
            dataset_id = message.get("dataset_id")
            if dataset_id:
                await self.stop_dataset_monitoring(dataset_id, client_id)
        
        elif message_type == "real_time_analysis":
            await self.handle_real_time_analysis(client_id, message)
    
    async def start_dataset_monitoring(self, dataset_id: str, client_id: str):
        """Start monitoring a dataset for changes"""
        if dataset_id not in self.monitoring_tasks:
            # Create monitoring task
            task = asyncio.create_task(self.monitor_dataset(dataset_id))
            self.monitoring_tasks[dataset_id] = task
        
        # Subscribe client to dataset updates
        self.connection_manager.subscribe(client_id, f"dataset_{dataset_id}")
        
        await self.connection_manager.send_personal_message({
            "type": "monitoring_started",
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat()
        }, client_id)
    
    async def stop_dataset_monitoring(self, dataset_id: str, client_id: str):
        """Stop monitoring a dataset"""
        # Unsubscribe client from dataset updates
        self.connection_manager.unsubscribe(client_id, f"dataset_{dataset_id}")
        
        await self.connection_manager.send_personal_message({
            "type": "monitoring_stopped",
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat()
        }, client_id)
    
    async def monitor_dataset(self, dataset_id: str):
        """Monitor a dataset for changes and send updates"""
        try:
            while True:
                # Get current dataset state
                df = self.file_service.get_dataframe(dataset_id)
                if df is not None:
                    # Calculate real-time statistics
                    stats = await self.calculate_real_time_stats(df)
                    
                    # Broadcast update to subscribers
                    await self.connection_manager.broadcast({
                        "type": "dataset_update",
                        "dataset_id": dataset_id,
                        "timestamp": datetime.now().isoformat(),
                        "stats": stats
                    }, f"dataset_{dataset_id}")
                
                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring stopped for dataset {dataset_id}")
        except Exception as e:
            logger.error(f"Error monitoring dataset {dataset_id}: {e}")
    
    async def calculate_real_time_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate real-time statistics for a dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "last_updated": datetime.now().isoformat()
        }
        
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols]
            stats["numeric_stats"] = {
                "mean": numeric_df.mean().to_dict(),
                "std": numeric_df.std().to_dict(),
                "min": numeric_df.min().to_dict(),
                "max": numeric_df.max().to_dict()
            }
        
        return stats
    
    async def handle_real_time_analysis(self, client_id: str, message: Dict[str, Any]):
        """Handle real-time analysis requests"""
        try:
            dataset_id = message.get("dataset_id")
            analysis_type = message.get("analysis_type")
            columns = message.get("columns", [])
            parameters = message.get("parameters", {})
            
            if not dataset_id or not analysis_type:
                await self.connection_manager.send_personal_message({
                    "type": "analysis_error",
                    "error": "Missing dataset_id or analysis_type",
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                return
            
            # Send progress update
            await self.connection_manager.send_personal_message({
                "type": "analysis_progress",
                "progress": 25,
                "message": "Starting analysis...",
                "timestamp": datetime.now().isoformat()
            }, client_id)

            # Perform analysis (these are sync methods, so we run them in a thread)
            import asyncio
            loop = asyncio.get_event_loop()

            try:
                if analysis_type == "descriptive":
                    result = await loop.run_in_executor(None,
                        self.analysis_service.descriptive_analysis, dataset_id, columns, parameters)
                elif analysis_type == "correlation":
                    result = await loop.run_in_executor(None,
                        self.analysis_service.correlation_analysis, dataset_id, columns, parameters)
                elif analysis_type == "anomaly_detection":
                    result = await loop.run_in_executor(None,
                        self.analysis_service.anomaly_detection, dataset_id, columns, parameters)
                else:
                    await self.connection_manager.send_personal_message({
                        "type": "analysis_error",
                        "error": f"Unsupported analysis type: {analysis_type}",
                        "timestamp": datetime.now().isoformat()
                    }, client_id)
                    return

                # Send progress update
                await self.connection_manager.send_personal_message({
                    "type": "analysis_progress",
                    "progress": 100,
                    "message": "Analysis complete!",
                    "timestamp": datetime.now().isoformat()
                }, client_id)

            except Exception as analysis_error:
                await self.connection_manager.send_personal_message({
                    "type": "analysis_error",
                    "error": str(analysis_error),
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                return
            
            # Send results
            await self.connection_manager.send_personal_message({
                "type": "analysis_complete",
                "analysis_type": analysis_type,
                "results": result if isinstance(result, dict) else result.__dict__,
                "timestamp": datetime.now().isoformat()
            }, client_id)

        except Exception as e:
            await self.connection_manager.send_personal_message({
                "type": "analysis_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, client_id)
    
    async def broadcast_system_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast system-wide updates"""
        await self.connection_manager.broadcast({
            "type": "system_update",
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }, "system_updates")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": self.connection_manager.get_connection_count(),
            "active_connections": self.connection_manager.get_connection_count(),
            "monitoring_tasks": len(self.monitoring_tasks),
            "streaming_datasets": len(self.streaming_datasets),
            "uptime_seconds": (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        }
