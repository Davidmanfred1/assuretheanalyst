"""
Enterprise Service
Handles audit logging, security, compliance, and enterprise integrations
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AuditEventType(str, Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    DATA_UPLOADED = "data_uploaded"
    DATA_DOWNLOADED = "data_downloaded"
    DATA_DELETED = "data_deleted"
    ANALYSIS_CREATED = "analysis_created"
    ANALYSIS_EXECUTED = "analysis_executed"
    REPORT_GENERATED = "report_generated"
    SHARE_LINK_CREATED = "share_link_created"
    SHARE_LINK_ACCESSED = "share_link_accessed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    SYSTEM_CONFIG_CHANGED = "system_config_changed"
    SECURITY_VIOLATION = "security_violation"

class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStandard(str, Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"

class AuditEvent:
    """Audit event model"""
    
    def __init__(self, event_type: AuditEventType, user_id: str, details: Dict[str, Any]):
        self.event_id = str(uuid.uuid4())
        self.event_type = event_type
        self.user_id = user_id
        self.timestamp = datetime.now()
        self.details = details
        self.ip_address = details.get("ip_address", "unknown")
        self.user_agent = details.get("user_agent", "unknown")
        self.session_id = details.get("session_id", "unknown")
        self.risk_score = self._calculate_risk_score()
    
    def _calculate_risk_score(self) -> int:
        """Calculate risk score for the event (0-100)"""
        base_scores = {
            AuditEventType.USER_LOGIN: 10,
            AuditEventType.USER_LOGOUT: 5,
            AuditEventType.USER_CREATED: 30,
            AuditEventType.USER_DELETED: 80,
            AuditEventType.DATA_UPLOADED: 40,
            AuditEventType.DATA_DELETED: 70,
            AuditEventType.PERMISSION_GRANTED: 50,
            AuditEventType.SECURITY_VIOLATION: 90,
            AuditEventType.SYSTEM_CONFIG_CHANGED: 60
        }
        
        base_score = base_scores.get(self.event_type, 20)
        
        # Adjust based on details
        if self.details.get("failed_attempt"):
            base_score += 20
        if self.details.get("admin_action"):
            base_score += 15
        if self.details.get("bulk_operation"):
            base_score += 10
        
        return min(base_score, 100)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "risk_score": self.risk_score
        }

class SecurityPolicy:
    """Security policy configuration"""
    
    def __init__(self):
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special_chars": True,
            "max_age_days": 90,
            "history_count": 5
        }
        
        self.session_policy = {
            "max_duration_hours": 8,
            "idle_timeout_minutes": 30,
            "max_concurrent_sessions": 3,
            "require_2fa": False
        }
        
        self.access_policy = {
            "max_failed_attempts": 5,
            "lockout_duration_minutes": 15,
            "ip_whitelist": [],
            "ip_blacklist": [],
            "require_vpn": False
        }
        
        self.data_policy = {
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "data_retention_days": 365,
            "backup_frequency_hours": 24,
            "anonymization_required": False
        }

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.limits = {
            "default": {"requests": 100, "window_minutes": 15},
            "upload": {"requests": 10, "window_minutes": 60},
            "analysis": {"requests": 50, "window_minutes": 60},
            "auth": {"requests": 5, "window_minutes": 15}
        }
        self.request_history: Dict[str, List[datetime]] = {}
    
    async def check_rate_limit(self, identifier: str, endpoint_type: str = "default") -> bool:
        """Check if request is within rate limits"""
        now = datetime.now()
        limit_config = self.limits.get(endpoint_type, self.limits["default"])
        
        window_start = now - timedelta(minutes=limit_config["window_minutes"])
        
        # Clean old requests
        if identifier in self.request_history:
            self.request_history[identifier] = [
                req_time for req_time in self.request_history[identifier]
                if req_time > window_start
            ]
        else:
            self.request_history[identifier] = []
        
        # Check limit
        if len(self.request_history[identifier]) >= limit_config["requests"]:
            return False
        
        # Add current request
        self.request_history[identifier].append(now)
        return True
    
    def get_rate_limit_status(self, identifier: str, endpoint_type: str = "default") -> Dict[str, Any]:
        """Get current rate limit status"""
        limit_config = self.limits.get(endpoint_type, self.limits["default"])
        current_requests = len(self.request_history.get(identifier, []))
        
        return {
            "limit": limit_config["requests"],
            "window_minutes": limit_config["window_minutes"],
            "current_requests": current_requests,
            "remaining": max(0, limit_config["requests"] - current_requests),
            "reset_time": (datetime.now() + timedelta(minutes=limit_config["window_minutes"])).isoformat()
        }

class EnterpriseService:
    """Enterprise features service"""
    
    def __init__(self):
        self.audit_log: List[AuditEvent] = []
        self.security_policy = SecurityPolicy()
        self.rate_limiter = RateLimiter()
        self.compliance_settings = {}
        self.integrations = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # Initialize compliance settings
        self._initialize_compliance()
        
        # Background tasks will be started when needed
        self._monitoring_started = False
    
    def _initialize_compliance(self):
        """Initialize compliance settings"""
        self.compliance_settings = {
            ComplianceStandard.GDPR: {
                "enabled": False,
                "data_retention_days": 365,
                "consent_required": True,
                "right_to_deletion": True,
                "data_portability": True
            },
            ComplianceStandard.HIPAA: {
                "enabled": False,
                "encryption_required": True,
                "access_logging": True,
                "minimum_necessary": True,
                "business_associate_agreement": False
            },
            ComplianceStandard.SOX: {
                "enabled": False,
                "financial_controls": True,
                "audit_trail": True,
                "segregation_of_duties": True,
                "change_management": True
            }
        }
    
    # Audit Logging
    async def log_audit_event(self, event_type: AuditEventType, user_id: str, 
                            details: Dict[str, Any]) -> str:
        """Log an audit event"""
        event = AuditEvent(event_type, user_id, details)
        self.audit_log.append(event)
        
        # Keep only last 10,000 events in memory
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
        
        # Check for security violations
        if event.risk_score > 70:
            await self._create_security_alert(event)
        
        logger.info(f"Audit event logged: {event_type} by user {user_id}")
        return event.event_id
    
    async def get_audit_log(self, user_id: str = None, event_type: AuditEventType = None,
                          start_date: datetime = None, end_date: datetime = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log with filters"""
        filtered_events = self.audit_log.copy()
        
        # Apply filters
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if start_date:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
        
        if end_date:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        filtered_events = filtered_events[:limit]
        
        return [event.to_dict() for event in filtered_events]
    
    # Security Management
    async def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against security policy"""
        policy = self.security_policy.password_policy
        issues = []
        
        if len(password) < policy["min_length"]:
            issues.append(f"Password must be at least {policy['min_length']} characters")
        
        if policy["require_uppercase"] and not any(c.isupper() for c in password):
            issues.append("Password must contain uppercase letters")
        
        if policy["require_lowercase"] and not any(c.islower() for c in password):
            issues.append("Password must contain lowercase letters")
        
        if policy["require_numbers"] and not any(c.isdigit() for c in password):
            issues.append("Password must contain numbers")
        
        if policy["require_special_chars"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain special characters")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "strength_score": self._calculate_password_strength(password)
        }
    
    def _calculate_password_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)"""
        score = 0
        
        # Length bonus
        score += min(password.__len__() * 4, 40)
        
        # Character variety bonus
        if any(c.isupper() for c in password):
            score += 10
        if any(c.islower() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 15
        
        # Complexity bonus
        if len(set(password)) > len(password) * 0.7:  # High character diversity
            score += 15
        
        return min(score, 100)
    
    async def check_security_violation(self, user_id: str, action: str, 
                                     context: Dict[str, Any]) -> bool:
        """Check for potential security violations"""
        violations = []
        
        # Check for suspicious patterns
        recent_events = [
            e for e in self.audit_log
            if e.user_id == user_id and 
            e.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        # Too many failed attempts
        failed_attempts = len([
            e for e in recent_events
            if e.details.get("failed_attempt", False)
        ])
        
        if failed_attempts > 5:
            violations.append("Too many failed attempts")
        
        # Unusual access patterns
        if len(recent_events) > 50:  # Too many actions in short time
            violations.append("Unusual activity volume")
        
        # Geographic anomalies (simplified)
        ip_address = context.get("ip_address", "")
        if ip_address and self._is_suspicious_ip(ip_address):
            violations.append("Suspicious IP address")
        
        if violations:
            await self.log_audit_event(
                AuditEventType.SECURITY_VIOLATION,
                user_id,
                {
                    "action": action,
                    "violations": violations,
                    "context": context
                }
            )
            return True
        
        return False
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious (simplified)"""
        # In a real implementation, this would check against threat intelligence
        blacklisted_ranges = ["192.168.1.100"]  # Example
        return ip_address in blacklisted_ranges
    
    # Rate Limiting
    async def check_rate_limit(self, identifier: str, endpoint_type: str = "default") -> bool:
        """Check rate limit for identifier"""
        return await self.rate_limiter.check_rate_limit(identifier, endpoint_type)
    
    def get_rate_limit_status(self, identifier: str, endpoint_type: str = "default") -> Dict[str, Any]:
        """Get rate limit status"""
        return self.rate_limiter.get_rate_limit_status(identifier, endpoint_type)
    
    # Compliance Management
    async def enable_compliance_standard(self, standard: ComplianceStandard, 
                                       settings: Dict[str, Any] = None) -> bool:
        """Enable compliance standard"""
        if standard in self.compliance_settings:
            self.compliance_settings[standard]["enabled"] = True
            
            if settings:
                self.compliance_settings[standard].update(settings)
            
            await self.log_audit_event(
                AuditEventType.SYSTEM_CONFIG_CHANGED,
                "system",
                {"compliance_standard": standard, "action": "enabled"}
            )
            
            return True
        
        return False
    
    async def check_compliance(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Check compliance status for a standard"""
        if standard not in self.compliance_settings:
            return {"compliant": False, "reason": "Standard not configured"}
        
        config = self.compliance_settings[standard]
        
        if not config["enabled"]:
            return {"compliant": False, "reason": "Standard not enabled"}
        
        # Perform compliance checks based on standard
        if standard == ComplianceStandard.GDPR:
            return await self._check_gdpr_compliance()
        elif standard == ComplianceStandard.HIPAA:
            return await self._check_hipaa_compliance()
        elif standard == ComplianceStandard.SOX:
            return await self._check_sox_compliance()
        
        return {"compliant": True, "checks_passed": []}
    
    async def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance"""
        checks = []
        issues = []
        
        # Check data retention
        if self.security_policy.data_policy["data_retention_days"] <= 365:
            checks.append("Data retention policy compliant")
        else:
            issues.append("Data retention period too long")
        
        # Check encryption
        if self.security_policy.data_policy["encryption_at_rest"]:
            checks.append("Data encryption at rest enabled")
        else:
            issues.append("Data encryption at rest not enabled")
        
        return {
            "compliant": len(issues) == 0,
            "checks_passed": checks,
            "issues": issues
        }
    
    async def _check_hipaa_compliance(self) -> Dict[str, Any]:
        """Check HIPAA compliance"""
        checks = []
        issues = []
        
        # Check access logging
        if len(self.audit_log) > 0:
            checks.append("Access logging enabled")
        else:
            issues.append("Access logging not properly configured")
        
        # Check encryption
        if (self.security_policy.data_policy["encryption_at_rest"] and 
            self.security_policy.data_policy["encryption_in_transit"]):
            checks.append("Encryption requirements met")
        else:
            issues.append("Encryption requirements not met")
        
        return {
            "compliant": len(issues) == 0,
            "checks_passed": checks,
            "issues": issues
        }
    
    async def _check_sox_compliance(self) -> Dict[str, Any]:
        """Check SOX compliance"""
        checks = []
        issues = []
        
        # Check audit trail
        if len(self.audit_log) > 0:
            checks.append("Audit trail maintained")
        else:
            issues.append("Audit trail not properly maintained")
        
        # Check change management
        config_changes = [
            e for e in self.audit_log
            if e.event_type == AuditEventType.SYSTEM_CONFIG_CHANGED
        ]
        
        if config_changes:
            checks.append("Change management tracking enabled")
        else:
            issues.append("Change management tracking not enabled")
        
        return {
            "compliant": len(issues) == 0,
            "checks_passed": checks,
            "issues": issues
        }
    
    # Alert Management
    async def _create_security_alert(self, event: AuditEvent):
        """Create security alert for high-risk events"""
        alert = {
            "alert_id": str(uuid.uuid4()),
            "type": "security",
            "severity": self._get_severity_from_risk_score(event.risk_score),
            "title": f"Security Event: {event.event_type}",
            "description": f"High-risk event detected for user {event.user_id}",
            "event_id": event.event_id,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        logger.warning(f"Security alert created: {alert['title']}")
    
    def _get_severity_from_risk_score(self, risk_score: int) -> str:
        """Convert risk score to severity level"""
        if risk_score >= 90:
            return "critical"
        elif risk_score >= 70:
            return "high"
        elif risk_score >= 50:
            return "medium"
        else:
            return "low"
    
    # Background Tasks
    async def _security_monitoring_loop(self):
        """Background security monitoring"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check for suspicious patterns
                await self._analyze_security_patterns()
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
    
    async def _compliance_check_loop(self):
        """Background compliance checking"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Run compliance checks for enabled standards
                for standard, config in self.compliance_settings.items():
                    if config["enabled"]:
                        result = await self.check_compliance(standard)
                        if not result["compliant"]:
                            logger.warning(f"Compliance violation detected for {standard}")
                
            except Exception as e:
                logger.error(f"Compliance checking error: {e}")
    
    async def _analyze_security_patterns(self):
        """Analyze audit log for security patterns"""
        # This is a simplified implementation
        # In production, this would use more sophisticated analysis
        
        recent_events = [
            e for e in self.audit_log
            if e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        # Check for brute force attempts
        failed_logins = [
            e for e in recent_events
            if e.event_type == AuditEventType.USER_LOGIN and 
            e.details.get("failed_attempt", False)
        ]
        
        if len(failed_logins) > 20:  # More than 20 failed logins in 24h
            await self._create_security_alert(AuditEvent(
                AuditEventType.SECURITY_VIOLATION,
                "system",
                {"pattern": "brute_force_detected", "count": len(failed_logins)}
            ))
    
    # Public API Methods
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        recent_events = [
            e for e in self.audit_log
            if e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_events_24h": len(recent_events),
            "high_risk_events_24h": len([e for e in recent_events if e.risk_score > 70]),
            "failed_logins_24h": len([
                e for e in recent_events
                if e.event_type == AuditEventType.USER_LOGIN and 
                e.details.get("failed_attempt", False)
            ]),
            "active_alerts": len([a for a in self.alerts if not a["acknowledged"]]),
            "compliance_status": {
                standard: config["enabled"]
                for standard, config in self.compliance_settings.items()
            }
        }
    
    def get_alerts(self, acknowledged: bool = None) -> List[Dict[str, Any]]:
        """Get security alerts"""
        alerts = self.alerts.copy()
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a["acknowledged"] == acknowledged]
        
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge a security alert"""
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_by"] = user_id
                alert["acknowledged_at"] = datetime.now().isoformat()
                
                await self.log_audit_event(
                    AuditEventType.SYSTEM_CONFIG_CHANGED,
                    user_id,
                    {"action": "alert_acknowledged", "alert_id": alert_id}
                )
                
                return True
        
        return False
