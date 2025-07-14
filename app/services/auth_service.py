"""
Authentication and Authorization Service
Handles user management, authentication, and access control
"""

import hashlib
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import json
from pathlib import Path

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(str, Enum):
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    CREATE_ANALYSIS = "create_analysis"
    SHARE_ANALYSIS = "share_analysis"
    MANAGE_USERS = "manage_users"
    ADMIN_ACCESS = "admin_access"

class User:
    """User model"""
    
    def __init__(self, user_id: str, username: str, email: str, role: UserRole):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.created_at = datetime.now()
        self.last_login = None
        self.is_active = True
        self.profile = {}
        self.preferences = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "profile": self.profile,
            "preferences": self.preferences
        }

class AuthService:
    """Authentication and authorization service"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.user_credentials: Dict[str, str] = {}  # username -> hashed_password
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.secret_key = "your-secret-key-change-in-production"
        self.token_expiry_hours = 24
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.DELETE_DATA,
                Permission.CREATE_ANALYSIS, Permission.SHARE_ANALYSIS,
                Permission.MANAGE_USERS, Permission.ADMIN_ACCESS
            ],
            UserRole.ANALYST: [
                Permission.READ_DATA, Permission.WRITE_DATA,
                Permission.CREATE_ANALYSIS, Permission.SHARE_ANALYSIS
            ],
            UserRole.VIEWER: [
                Permission.READ_DATA
            ],
            UserRole.GUEST: []
        }
        
        # Create default admin user
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for demo purposes"""
        # Admin user
        admin_id = str(uuid.uuid4())
        admin_user = User(admin_id, "admin", "admin@assuretheanalyst.com", UserRole.ADMIN)
        self.users[admin_id] = admin_user
        self.user_credentials["admin"] = self._hash_password("admin123")
        
        # Analyst user
        analyst_id = str(uuid.uuid4())
        analyst_user = User(analyst_id, "analyst", "analyst@assuretheanalyst.com", UserRole.ANALYST)
        self.users[analyst_id] = analyst_user
        self.user_credentials["analyst"] = self._hash_password("analyst123")
        
        # Viewer user
        viewer_id = str(uuid.uuid4())
        viewer_user = User(viewer_id, "viewer", "viewer@assuretheanalyst.com", UserRole.VIEWER)
        self.users[viewer_id] = viewer_user
        self.user_credentials["viewer"] = self._hash_password("viewer123")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self._hash_password(password) == hashed_password
    
    def _generate_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def register_user(self, username: str, email: str, password: str, role: UserRole = UserRole.VIEWER) -> Dict[str, Any]:
        """Register a new user"""
        # Check if username already exists
        if username in self.user_credentials:
            return {"success": False, "message": "Username already exists"}
        
        # Check if email already exists
        for user in self.users.values():
            if user.email == email:
                return {"success": False, "message": "Email already exists"}
        
        # Create new user
        user_id = str(uuid.uuid4())
        user = User(user_id, username, email, role)
        
        self.users[user_id] = user
        self.user_credentials[username] = self._hash_password(password)
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user": user.to_dict()
        }
    
    async def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return token"""
        if username not in self.user_credentials:
            return {"success": False, "message": "Invalid username or password"}
        
        if not self._verify_password(password, self.user_credentials[username]):
            return {"success": False, "message": "Invalid username or password"}
        
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user or not user.is_active:
            return {"success": False, "message": "User account is inactive"}
        
        # Update last login
        user.last_login = datetime.now()
        
        # Generate token
        token = self._generate_token(user)
        
        # Store session
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "user_id": user.user_id,
            "token": token,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        return {
            "success": True,
            "message": "Authentication successful",
            "token": token,
            "session_id": session_id,
            "user": user.to_dict()
        }
    
    async def verify_session(self, token: str) -> Optional[User]:
        """Verify session token and return user"""
        payload = self._verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get("user_id")
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        if not user.is_active:
            return None
        
        return user
    
    async def logout(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.role_permissions.get(user.role, [])
        return permission in user_permissions
    
    def require_permission(self, user: User, permission: Permission) -> bool:
        """Require user to have specific permission (raises exception if not)"""
        if not self.has_permission(user, permission):
            raise PermissionError(f"User {user.username} does not have permission: {permission}")
        return True
    
    async def get_users(self) -> List[Dict[str, Any]]:
        """Get all users (admin only)"""
        return [user.to_dict() for user in self.users.values()]
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific user"""
        if user_id in self.users:
            return self.users[user_id].to_dict()
        return None
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        if "email" in updates:
            user.email = updates["email"]
        if "role" in updates:
            user.role = UserRole(updates["role"])
        if "is_active" in updates:
            user.is_active = updates["is_active"]
        if "profile" in updates:
            user.profile.update(updates["profile"])
        if "preferences" in updates:
            user.preferences.update(updates["preferences"])
        
        return True
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Remove from credentials
        if user.username in self.user_credentials:
            del self.user_credentials[user.username]
        
        # Remove user
        del self.users[user_id]
        
        # Invalidate all sessions for this user
        sessions_to_remove = [
            session_id for session_id, session in self.active_sessions.items()
            if session["user_id"] == user_id
        ]
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        return True
    
    async def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        if username not in self.user_credentials:
            return False
        
        if not self._verify_password(old_password, self.user_credentials[username]):
            return False
        
        self.user_credentials[username] = self._hash_password(new_password)
        return True
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions"""
        sessions = []
        for session_id, session in self.active_sessions.items():
            user = self.users.get(session["user_id"])
            if user:
                sessions.append({
                    "session_id": session_id,
                    "user": user.to_dict(),
                    "created_at": session["created_at"].isoformat(),
                    "last_activity": session["last_activity"].isoformat()
                })
        return sessions
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            # Check if session is older than token expiry
            if current_time - session["created_at"] > timedelta(hours=self.token_expiry_hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        return len(expired_sessions)
