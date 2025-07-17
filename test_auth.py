#!/usr/bin/env python3
"""
Test script to verify HuggingFace authentication functionality
"""

import sys
from pathlib import Path

# Add the huggingdrive directory to the path
sys.path.insert(0, str(Path(__file__).parent / "huggingdrive"))

from auth_manager import HuggingFaceAuthManager

def test_auth_manager():
    """Test the authentication manager"""
    
    print("Testing HuggingFace Authentication Manager...")
    
    # Create auth manager
    auth_manager = HuggingFaceAuthManager()
    
    # Check initial state
    print(f"Initial authentication status: {auth_manager.is_authenticated()}")
    print(f"Current username: {auth_manager.get_username()}")
    
    # Test with invalid token
    print("\nTesting with invalid token...")
    success, message = auth_manager.login("invalid_token")
    print(f"Login result: {success} - {message}")
    
    # Test logout
    print("\nTesting logout...")
    auth_manager.logout()
    print(f"After logout - authenticated: {auth_manager.is_authenticated()}")
    
    print("\nTest completed!")
    print("\nTo test with a real token:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Use the token in your HuggingDrive app")

if __name__ == "__main__":
    test_auth_manager() 