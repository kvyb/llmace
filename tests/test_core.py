"""Tests for core ACE functionality."""

import pytest
from llmace.core.context import ACEContext
from llmace.core.schemas import Bullet, BulletDelta, ContextConfig


class TestBullet:
    """Test Bullet class."""
    
    def test_bullet_creation(self):
        """Test creating a bullet."""
        bullet = Bullet(
            id="test-001",
            section="strategies",
            content="This is a test strategy"
        )
        assert bullet.id == "test-001"
        assert bullet.section == "strategies"
        assert bullet.content == "This is a test strategy"
        assert bullet.helpful_count == 0
        assert bullet.harmful_count == 0
    
    def test_bullet_score(self):
        """Test bullet scoring."""
        bullet = Bullet(
            id="test-001",
            section="strategies",
            content="Test"
        )
        bullet.increment_helpful()
        bullet.increment_helpful()
        bullet.increment_harmful()
        
        assert bullet.helpful_count == 2
        assert bullet.harmful_count == 1
        assert bullet.get_score() == 1
    
    def test_bullet_validation(self):
        """Test bullet validation."""
        with pytest.raises(ValueError):
            Bullet(
                id="test-001",
                section="strategies",
                content=""  # Empty content should fail
            )


class TestACEContext:
    """Test ACEContext class."""
    
    def test_context_creation(self):
        """Test creating a context."""
        context = ACEContext()
        assert len(context) == 0
        assert len(context.config.sections) > 0
    
    def test_add_bullet(self):
        """Test adding bullets."""
        context = ACEContext()
        bullet = context.add_bullet(
            section="strategies",
            content="Test strategy"
        )
        
        assert len(context) == 1
        assert bullet.section == "strategies"
        assert bullet.content == "Test strategy"
    
    def test_get_bullets_by_section(self):
        """Test retrieving bullets by section."""
        context = ACEContext()
        context.add_bullet(section="strategies", content="Strategy 1")
        context.add_bullet(section="strategies", content="Strategy 2")
        context.add_bullet(section="insights", content="Insight 1")
        
        strategies = context.get_bullets_by_section("strategies")
        insights = context.get_bullets_by_section("insights")
        
        assert len(strategies) == 2
        assert len(insights) == 1
    
    def test_merge_delta(self):
        """Test merging delta updates."""
        context = ACEContext()
        delta = BulletDelta(
            operation="add",
            section="strategies",
            content="New strategy"
        )
        
        bullet = context.merge_delta(delta)
        assert len(context) == 1
        assert bullet.content == "New strategy"
    
    def test_serialization(self):
        """Test context serialization."""
        context = ACEContext()
        context.add_bullet(section="strategies", content="Test")
        
        # Serialize
        data = context.to_dict()
        assert "config" in data
        assert "bullets" in data
        
        # Deserialize
        context2 = ACEContext.from_dict(data)
        assert len(context2) == len(context)
        assert len(context2.get_bullets_by_section("strategies")) == 1


class TestContextConfig:
    """Test ContextConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ContextConfig()
        assert len(config.sections) > 0
        assert config.dedup_threshold > 0
        assert config.enable_deduplication is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextConfig(
            sections=["custom1", "custom2"],
            dedup_threshold=0.95,
            max_bullets_per_section=10
        )
        
        assert config.sections == ["custom1", "custom2"]
        assert config.dedup_threshold == 0.95
        assert config.max_bullets_per_section == 10

