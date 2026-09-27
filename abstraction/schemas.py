"""Pydantic schemas for the 5 chart-abstraction targets.

Kept intentionally narrow: 5 fields matched to Layer Health's public product surface.
Each field has a matching gold-label JSON structure for eval.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    description: str = Field(..., description="Full text of the diagnosis")
    icd10: str | None = Field(None, description="ICD-10 code if stated in the note")


class Medication(BaseModel):
    name: str = Field(..., description="Medication name (generic preferred)")
    dose: str = Field(..., description="Dose with units (e.g. '81 mg', '24 units')")
    route: str = Field(..., description="Route of administration (PO, IV, subcutaneous, inhalation, etc.)")
    frequency: str = Field(..., description="Frequency (daily, BID, q4h PRN, etc.)")


class Procedure(BaseModel):
    name: str = Field(..., description="Procedure name")
    date: str | None = Field(None, description="Date in ISO YYYY-MM-DD format if stated")


class LabValue(BaseModel):
    test: str = Field(..., description="Test name (e.g. 'Troponin I', 'HbA1c')")
    value: str = Field(..., description="Numeric value as a string")
    unit: str = Field(..., description="Units (mg/dL, %, etc.); empty string if unitless")
    date: str | None = Field(None, description="Date in ISO YYYY-MM-DD format if stated")


class TimelineEvent(BaseModel):
    date: str = Field(..., description="Date in ISO YYYY-MM-DD format")
    event: str = Field(..., description="Short description of the clinical event on that date")


class ChartAbstraction(BaseModel):
    """Container for the full structured extraction from a single clinical note."""
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    lab_values: list[LabValue] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)


SCHEMA_JSON_HINT = ChartAbstraction.model_json_schema()
