"""Tests for the JSON Data Layer."""

import os
import pytest
import shutil
from unittest.mock import patch
from chainlit.user import User
from chainlit.types import Pagination, ThreadFilter

from data_layer import JsonDataLayer

# Use a temporary directory for tests
TEST_ROOT = os.path.join(os.path.dirname(__file__), ".test_chat_data")

@pytest.fixture
def data_layer():
    """Setup and teardown the data layer with temporary storage."""
    # Setup
    if os.path.exists(TEST_ROOT):
        shutil.rmtree(TEST_ROOT)
    os.makedirs(TEST_ROOT)

    # Patch the paths in the data_layer module
    with patch("data_layer._ROOT", TEST_ROOT), \
         patch("data_layer._USERS_FILE", os.path.join(TEST_ROOT, "users.json")), \
         patch("data_layer._THREADS_DIR", os.path.join(TEST_ROOT, "threads")):
        
        dl = JsonDataLayer()
        yield dl

    # Teardown
    if os.path.exists(TEST_ROOT):
        shutil.rmtree(TEST_ROOT)

@pytest.mark.asyncio
async def test_create_and_get_user(data_layer):
    """Should create and retrieve a user."""
    user = User(identifier="test_user", metadata={"role": "admin"})
    
    # Create
    persisted_user = await data_layer.create_user(user)
    assert persisted_user is not None
    assert persisted_user.identifier == "test_user"
    assert persisted_user.metadata["role"] == "admin"

    # Get
    fetched_user = await data_layer.get_user("test_user")
    assert fetched_user is not None
    assert fetched_user.identifier == "test_user"
    assert fetched_user.metadata == {"role": "admin"}

@pytest.mark.asyncio
async def test_get_missing_user(data_layer):
    """Should return None for non-existent user."""
    user = await data_layer.get_user("non_existent")
    assert user is None

@pytest.mark.asyncio
async def test_thread_lifecycle(data_layer):
    """Should create, update, get, list and delete threads."""
    thread_id = "thread_123"
    
    # 1. Update (creates if not exists in this implementation's update_thread logic?)
    # Actually JsonDataLayer.update_thread creates if missing in lines 126-130
    await data_layer.update_thread(
        thread_id=thread_id,
        name="Test Thread",
        user_id="user_abc"
    )

    # 2. Get
    thread = await data_layer.get_thread(thread_id)
    assert thread is not None
    assert thread["id"] == thread_id
    assert thread["name"] == "Test Thread"
    assert thread["userIdentifier"] == "user_abc"

    # 3. List
    pagination = Pagination(first=10)
    filters = ThreadFilter(userId="user_abc")
    res = await data_layer.list_threads(pagination, filters)
    assert len(res.data) == 1
    assert res.data[0]["id"] == thread_id

    # 4. Delete
    await data_layer.delete_thread(thread_id)
    thread = await data_layer.get_thread(thread_id)
    assert thread is None

@pytest.mark.asyncio
async def test_step_lifecycle(data_layer):
    """Should create and persist steps."""
    thread_id = "thread_steps"
    step_id = "step_1"
    
    # Create valid thread first (although create_step creates if missing? No, create_step expects thread_id to ensure path)
    # create_step logic: lines 206-210 create a default dict if file missing.
    
    step_data = {
        "id": step_id,
        "threadId": thread_id,
        "name": "User",
        "type": "user_message",
        "content": "Hello"
    }
    
    await data_layer.create_step(step_data)
    
    thread = await data_layer.get_thread(thread_id)
    assert thread is not None
    assert len(thread["steps"]) == 1
    assert thread["steps"][0]["content"] == "Hello"

    # Update step
    step_data["content"] = "Hello Updated"
    await data_layer.update_step(step_data)
    
    thread = await data_layer.get_thread(thread_id)
    assert thread["steps"][0]["content"] == "Hello Updated"

    # Delete step
    await data_layer.delete_step(step_id)
    thread = await data_layer.get_thread(thread_id)
    assert len(thread["steps"]) == 0
