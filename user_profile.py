import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

PROFILE_DIR = Path(os.getenv("PROFILE_DIR", "user_profiles"))


class CoreSlots(BaseModel):
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    body_type: Optional[str] = None
    fuel_type: Optional[str] = None
    brand: Optional[str] = None
    condition: Optional[str] = None


class SoftPreferences(BaseModel):
    features: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None


class UserProfile(BaseModel):
    core_slots: CoreSlots = Field(default_factory=CoreSlots)
    soft_preferences: SoftPreferences = Field(default_factory=SoftPreferences)
    viewed_models: List[str] = Field(default_factory=list)
    excluded_brands: List[str] = Field(default_factory=list)


class ProfileUpdate(BaseModel):
    budget_min: Optional[float] = Field(default=None, description="Minimum budget as a USD number.")
    budget_max: Optional[float] = Field(default=None, description="Maximum budget as a USD number.")
    body_type: Optional[str] = Field(default=None, description="Desired body or usage type, e.g. SUV, Sedan.")
    fuel_type: Optional[str] = Field(default=None, description="Desired fuel type, e.g. Gasoline, Hybrid, Electric.")
    brand: Optional[str] = Field(default=None, description="Preferred brand if any.")
    condition: Optional[str] = Field(default=None, description="New or Used.")
    add_features: List[str] = Field(default_factory=list, description="New desired features to remember.")
    vibe: Optional[str] = Field(default=None, description="Overall vibe the customer wants, e.g. luxurious, sporty.")
    exclude_brands: List[str] = Field(default_factory=list, description="Brands the customer wants to avoid.")
    interested_models: List[str] = Field(default_factory=list, description="Specific models the customer asks about.")


def merge_update(profile: UserProfile, update: ProfileUpdate) -> UserProfile:
    cs = profile.core_slots
    for field in ("budget_min", "budget_max", "body_type", "fuel_type", "brand", "condition"):
        value = getattr(update, field)
        if value is not None:
            setattr(cs, field, value)

    sp = profile.soft_preferences
    for feature in update.add_features:
        if feature and feature not in sp.features:
            sp.features.append(feature)
    if update.vibe:
        sp.vibe = update.vibe

    for brand in update.exclude_brands:
        if brand and brand not in profile.excluded_brands:
            profile.excluded_brands.append(brand)
    for model in update.interested_models:
        if model and model not in profile.viewed_models:
            profile.viewed_models.append(model)
    return profile


def log_viewed(profile: UserProfile, titles: List[str]) -> UserProfile:
    for title in titles:
        if title and title not in profile.viewed_models:
            profile.viewed_models.append(title)
    return profile


def _path(session_id: str) -> Path:
    return PROFILE_DIR / f"{session_id}.json"


def load_profile(session_id: str) -> UserProfile:
    path = _path(session_id)
    if path.exists():
        try:
            return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return UserProfile()


def save_profile(session_id: str, profile: UserProfile) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    _path(session_id).write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def delete_profile(session_id: str) -> None:
    _path(session_id).unlink(missing_ok=True)
