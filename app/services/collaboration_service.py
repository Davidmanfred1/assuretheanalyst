"""
Collaboration Service
Handles projects, sharing, comments, and team collaboration
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from app.services.auth_service import User, UserRole

class ProjectRole(str, Enum):
    OWNER = "owner"
    COLLABORATOR = "collaborator"
    VIEWER = "viewer"

class SharePermission(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

class CommentType(str, Enum):
    GENERAL = "general"
    ANALYSIS = "analysis"
    DATASET = "dataset"
    CHART = "chart"

class Project:
    """Project model for organizing work"""
    
    def __init__(self, project_id: str, name: str, description: str, owner_id: str):
        self.project_id = project_id
        self.name = name
        self.description = description
        self.owner_id = owner_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.is_public = False
        self.tags = []
        self.datasets = []
        self.analyses = []
        self.charts = []
        self.members = {owner_id: ProjectRole.OWNER}
        self.settings = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_public": self.is_public,
            "tags": self.tags,
            "datasets": self.datasets,
            "analyses": self.analyses,
            "charts": self.charts,
            "members": self.members,
            "settings": self.settings
        }

class Comment:
    """Comment model for discussions"""
    
    def __init__(self, comment_id: str, user_id: str, content: str, 
                 comment_type: CommentType, target_id: str):
        self.comment_id = comment_id
        self.user_id = user_id
        self.content = content
        self.comment_type = comment_type
        self.target_id = target_id  # ID of the item being commented on
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.is_edited = False
        self.replies = []
        self.reactions = {}  # user_id -> reaction_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "comment_id": self.comment_id,
            "user_id": self.user_id,
            "content": self.content,
            "comment_type": self.comment_type,
            "target_id": self.target_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_edited": self.is_edited,
            "replies": self.replies,
            "reactions": self.reactions
        }

class ShareLink:
    """Shareable link model"""
    
    def __init__(self, link_id: str, resource_type: str, resource_id: str, 
                 created_by: str, permission: SharePermission):
        self.link_id = link_id
        self.resource_type = resource_type  # project, analysis, chart, etc.
        self.resource_id = resource_id
        self.created_by = created_by
        self.permission = permission
        self.created_at = datetime.now()
        self.expires_at = None
        self.is_active = True
        self.access_count = 0
        self.last_accessed = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "created_by": self.created_by,
            "permission": self.permission,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }

class CollaborationService:
    """Service for managing collaboration features"""
    
    def __init__(self):
        self.projects: Dict[str, Project] = {}
        self.comments: Dict[str, Comment] = {}
        self.share_links: Dict[str, ShareLink] = {}
        self.activity_log: List[Dict[str, Any]] = []
    
    # Project Management
    async def create_project(self, name: str, description: str, owner_id: str, 
                           tags: List[str] = None) -> str:
        """Create a new project"""
        project_id = str(uuid.uuid4())
        project = Project(project_id, name, description, owner_id)
        
        if tags:
            project.tags = tags
        
        self.projects[project_id] = project
        
        # Log activity
        await self._log_activity("project_created", owner_id, {
            "project_id": project_id,
            "project_name": name
        })
        
        return project_id
    
    async def get_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """Get projects accessible to user"""
        accessible_projects = []
        
        for project in self.projects.values():
            # Check if user is a member or project is public
            if user_id in project.members or project.is_public:
                accessible_projects.append(project.to_dict())
        
        return accessible_projects
    
    async def get_project(self, project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific project if user has access"""
        if project_id not in self.projects:
            return None
        
        project = self.projects[project_id]
        
        # Check access
        if user_id not in project.members and not project.is_public:
            return None
        
        return project.to_dict()
    
    async def update_project(self, project_id: str, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update project (owner/collaborator only)"""
        if project_id not in self.projects:
            return False
        
        project = self.projects[project_id]
        
        # Check permissions
        user_role = project.members.get(user_id)
        if user_role not in [ProjectRole.OWNER, ProjectRole.COLLABORATOR]:
            return False
        
        # Update fields
        if "name" in updates:
            project.name = updates["name"]
        if "description" in updates:
            project.description = updates["description"]
        if "tags" in updates:
            project.tags = updates["tags"]
        if "is_public" in updates and user_role == ProjectRole.OWNER:
            project.is_public = updates["is_public"]
        
        project.updated_at = datetime.now()
        
        # Log activity
        await self._log_activity("project_updated", user_id, {
            "project_id": project_id,
            "updates": list(updates.keys())
        })
        
        return True
    
    async def delete_project(self, project_id: str, user_id: str) -> bool:
        """Delete project (owner only)"""
        if project_id not in self.projects:
            return False
        
        project = self.projects[project_id]
        
        # Check if user is owner
        if project.members.get(user_id) != ProjectRole.OWNER:
            return False
        
        # Delete related comments
        comments_to_delete = [
            comment_id for comment_id, comment in self.comments.items()
            if comment.target_id == project_id
        ]
        for comment_id in comments_to_delete:
            del self.comments[comment_id]
        
        # Delete related share links
        links_to_delete = [
            link_id for link_id, link in self.share_links.items()
            if link.resource_id == project_id
        ]
        for link_id in links_to_delete:
            del self.share_links[link_id]
        
        del self.projects[project_id]
        
        # Log activity
        await self._log_activity("project_deleted", user_id, {
            "project_id": project_id
        })
        
        return True
    
    # Project Membership
    async def add_project_member(self, project_id: str, user_id: str, new_member_id: str, 
                               role: ProjectRole) -> bool:
        """Add member to project"""
        if project_id not in self.projects:
            return False
        
        project = self.projects[project_id]
        
        # Check if user is owner or collaborator
        user_role = project.members.get(user_id)
        if user_role not in [ProjectRole.OWNER, ProjectRole.COLLABORATOR]:
            return False
        
        # Only owner can add collaborators
        if role == ProjectRole.COLLABORATOR and user_role != ProjectRole.OWNER:
            return False
        
        project.members[new_member_id] = role
        project.updated_at = datetime.now()
        
        # Log activity
        await self._log_activity("member_added", user_id, {
            "project_id": project_id,
            "new_member_id": new_member_id,
            "role": role
        })
        
        return True
    
    async def remove_project_member(self, project_id: str, user_id: str, member_id: str) -> bool:
        """Remove member from project"""
        if project_id not in self.projects:
            return False
        
        project = self.projects[project_id]
        
        # Check if user is owner
        if project.members.get(user_id) != ProjectRole.OWNER:
            return False
        
        # Cannot remove owner
        if project.members.get(member_id) == ProjectRole.OWNER:
            return False
        
        if member_id in project.members:
            del project.members[member_id]
            project.updated_at = datetime.now()
            
            # Log activity
            await self._log_activity("member_removed", user_id, {
                "project_id": project_id,
                "removed_member_id": member_id
            })
            
            return True
        
        return False
    
    # Comments System
    async def add_comment(self, user_id: str, content: str, comment_type: CommentType, 
                         target_id: str, parent_comment_id: str = None) -> str:
        """Add a comment"""
        comment_id = str(uuid.uuid4())
        comment = Comment(comment_id, user_id, content, comment_type, target_id)
        
        if parent_comment_id and parent_comment_id in self.comments:
            # Add as reply
            parent_comment = self.comments[parent_comment_id]
            parent_comment.replies.append(comment_id)
        
        self.comments[comment_id] = comment
        
        # Log activity
        await self._log_activity("comment_added", user_id, {
            "comment_id": comment_id,
            "target_id": target_id,
            "comment_type": comment_type
        })
        
        return comment_id
    
    async def get_comments(self, target_id: str, comment_type: CommentType = None) -> List[Dict[str, Any]]:
        """Get comments for a target"""
        comments = []
        
        for comment in self.comments.values():
            if comment.target_id == target_id:
                if comment_type is None or comment.comment_type == comment_type:
                    comments.append(comment.to_dict())
        
        # Sort by creation date
        comments.sort(key=lambda x: x["created_at"])
        return comments
    
    async def update_comment(self, comment_id: str, user_id: str, content: str) -> bool:
        """Update comment (author only)"""
        if comment_id not in self.comments:
            return False
        
        comment = self.comments[comment_id]
        
        if comment.user_id != user_id:
            return False
        
        comment.content = content
        comment.updated_at = datetime.now()
        comment.is_edited = True
        
        return True
    
    async def delete_comment(self, comment_id: str, user_id: str) -> bool:
        """Delete comment (author only)"""
        if comment_id not in self.comments:
            return False
        
        comment = self.comments[comment_id]
        
        if comment.user_id != user_id:
            return False
        
        # Remove from parent's replies if it's a reply
        for parent_comment in self.comments.values():
            if comment_id in parent_comment.replies:
                parent_comment.replies.remove(comment_id)
        
        del self.comments[comment_id]
        return True
    
    # Sharing System
    async def create_share_link(self, resource_type: str, resource_id: str, 
                              user_id: str, permission: SharePermission, 
                              expires_at: datetime = None) -> str:
        """Create a shareable link"""
        link_id = str(uuid.uuid4())
        share_link = ShareLink(link_id, resource_type, resource_id, user_id, permission)
        
        if expires_at:
            share_link.expires_at = expires_at
        
        self.share_links[link_id] = share_link
        
        # Log activity
        await self._log_activity("share_link_created", user_id, {
            "link_id": link_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "permission": permission
        })
        
        return link_id
    
    async def access_share_link(self, link_id: str) -> Optional[Dict[str, Any]]:
        """Access a shared resource via link"""
        if link_id not in self.share_links:
            return None
        
        share_link = self.share_links[link_id]
        
        # Check if link is active
        if not share_link.is_active:
            return None
        
        # Check if link has expired
        if share_link.expires_at and datetime.now() > share_link.expires_at:
            share_link.is_active = False
            return None
        
        # Update access statistics
        share_link.access_count += 1
        share_link.last_accessed = datetime.now()
        
        return {
            "resource_type": share_link.resource_type,
            "resource_id": share_link.resource_id,
            "permission": share_link.permission,
            "share_info": share_link.to_dict()
        }
    
    async def get_share_links(self, user_id: str) -> List[Dict[str, Any]]:
        """Get share links created by user"""
        user_links = []
        
        for link in self.share_links.values():
            if link.created_by == user_id:
                user_links.append(link.to_dict())
        
        return user_links
    
    async def revoke_share_link(self, link_id: str, user_id: str) -> bool:
        """Revoke a share link"""
        if link_id not in self.share_links:
            return False
        
        share_link = self.share_links[link_id]
        
        if share_link.created_by != user_id:
            return False
        
        share_link.is_active = False
        return True
    
    # Activity Logging
    async def _log_activity(self, action: str, user_id: str, details: Dict[str, Any]):
        """Log user activity"""
        activity = {
            "activity_id": str(uuid.uuid4()),
            "action": action,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        self.activity_log.append(activity)
        
        # Keep only last 1000 activities
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-1000:]
    
    async def get_activity_log(self, user_id: str = None, project_id: str = None, 
                             limit: int = 50) -> List[Dict[str, Any]]:
        """Get activity log"""
        activities = self.activity_log.copy()
        
        # Filter by user
        if user_id:
            activities = [a for a in activities if a["user_id"] == user_id]
        
        # Filter by project
        if project_id:
            activities = [
                a for a in activities 
                if a["details"].get("project_id") == project_id
            ]
        
        # Sort by timestamp (newest first) and limit
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        return activities[:limit]
    
    # Utility Methods
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics"""
        return {
            "total_projects": len(self.projects),
            "total_comments": len(self.comments),
            "total_share_links": len(self.share_links),
            "active_share_links": len([l for l in self.share_links.values() if l.is_active]),
            "total_activities": len(self.activity_log)
        }
