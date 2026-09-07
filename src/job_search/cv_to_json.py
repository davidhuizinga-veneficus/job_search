"""Convert a PDF CV into the application's structured JSON format."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pypdf import PdfReader

load_dotenv()  # Load environment variables from .env file


class ProfessionalProfile(BaseModel):
    current_job_title: str = ""
    professional_summary: str = ""
    years_of_experience: float | None = None
    industries: list[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    job_title: str = ""
    company: str = ""
    employment_type: str = ""
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None
    currently_employed: bool = False
    industry: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class Education(BaseModel):
    degree: str = ""
    field_of_study: str = ""
    level: str = ""
    institution: str = ""
    graduation_date: datetime.date | None = None


class Language(BaseModel):
    language: str = ""
    proficiency: str = ""


class Certification(BaseModel):
    name: str = ""
    issuing_organization: str = ""
    expiry_date: datetime.date | None = None


class Project(BaseModel):
    name: str = ""
    description: str = ""
    skills: list[str] = Field(default_factory=list)


class WorkAuthorization(BaseModel):
    countries: list[str] = Field(default_factory=list)
    requires_sponsorship: bool | None = None


class Location(BaseModel):
    country: str = ""
    region: str = ""
    city: str = ""


class CV(BaseModel):
    """The JSON contract produced from a CV."""

    model_config = ConfigDict(extra="ignore")

    professional_profile: ProfessionalProfile = Field(
        default_factory=ProfessionalProfile
    )

    work_experience: list[WorkExperience] = Field(
        default_factory=list
    )

    education: list[Education] = Field(
        default_factory=list
    )

    skills: list[str] = Field(
        default_factory=list
    )

    languages: list[Language] = Field(
        default_factory=list
    )

    certifications: list[Certification] = Field(
        default_factory=list
    )

    projects: list[Project] = Field(
        default_factory=list
    )

    work_authorization: WorkAuthorization = Field(
        default_factory=WorkAuthorization
    )

    location: Location = Field(
        default_factory=Location
    )


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract text from the PDF."""

    path = Path(pdf_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("The CV file must be a PDF.")

    if not path.is_file():
        raise FileNotFoundError(f"CV file not found: {path}")

    reader = PdfReader(str(path))

    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    text = "\n\n".join(pages).strip()

    if not text:
        raise ValueError("No text could be extracted from the PDF.")

    return text


def calculate_days(
    start_date: datetime.date,
    end_date: datetime.date,
) -> int:
    """Calculate the number of days between two dates."""
    return (end_date - start_date).days


def calculate_years_of_experience(
    work_experience: list[WorkExperience],
) -> float | None:
    """Calculate years of experience from the overall employment period."""

    start_dates = [
        experience.start_date
        for experience in work_experience
        if experience.start_date is not None
    ]

    if not start_dates:
        return None

    end_dates = [
        experience.end_date
        for experience in work_experience
        if experience.end_date is not None
    ]

    # Currently employed positions are considered to continue until today.
    for experience in work_experience:
        if experience.currently_employed:
            end_dates.append(datetime.date.today())

    if not end_dates:
        return None

    earliest_start = min(start_dates)
    latest_end = max(end_dates)

    days = calculate_days(
        earliest_start,
        latest_end,
    )

    # Convert days to years.
    # 365.25 accounts approximately for leap years.
    years = days / 365.25

    return round(years, 2)


def convert_cv_to_json(
    pdf_path: str | Path,
    output_path: str | Path,
    api_key: str | None = None,
) -> CV:
    """Extract a PDF CV and write its AI-structured representation as JSON."""

    path = Path(pdf_path)

    # Extract text from PDF
    extracted_text = extract_pdf_text(path)

    # Get Google API key
    google_api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError(
            "Provide api_key or set the GOOGLE_API_KEY environment variable."
        )

    # Configure Gemini
    provider = GoogleProvider(api_key=google_api_key)

    model = GoogleModel(
        "gemini-flash-lite-latest",
        provider=provider,
    )

    # Create structured extraction agent
    agent = Agent(
        model,
        output_type=CV, 
		instructions=(
		"Extract all relevant information from the CV text into the provided CV schema. "
		"Never omit a work experience, education entry, certification, project, language, "
		"skill, or other relevant item that is explicitly present in the CV. "
		"For work experience, identify and extract every distinct employment position, "
		"including multiple positions at the same company when they are listed separately. "
		"For education, identify and extract every distinct formal education entry explicitly "
		"listed in the CV, such as degrees, diplomas, colleges, universities, academic programs, "
		"training programs, courses, and other educational qualifications. "
		"For certifications, identify and extract every certification, certificate, credential, "
		"license, qualification, professional training program, or specialized training explicitly "
		"listed in the CV, regardless of which section or heading it appears under. "
		"If a certification, certificate, credential, license, qualification, professional training "
		"program, or specialized training appears under an Education, Training, Qualifications, "
		"or similar heading, include it under certifications even if it is also included under "
		"education. "
		"It is acceptable and expected for an item to appear in both education and certifications "
		"when the item represents both an educational/training entry and a certification, "
		"certificate, credential, license, qualification, or professional certification. "
		"Do not remove an item from education merely because it is also classified as a certification. "
		"Do not remove an item from certifications merely because it appears under an Education "
		"heading. "
		"Classify each item based on the nature of the item itself, not only on the CV section "
		"heading where it appears. "
		"Do not stop extracting after finding the first few entries. "
		"Never invent information. Use empty strings, empty arrays, or null "
		"when a value is not present. "
		"Normalize dates to ISO format YYYY-MM-DD. "
		"If a date contains only a month and year, use the first day of that month. "
		"For example, if the CV says 'May 2020', use '2020-05-01'. "
		"If a date contains only a year, use the first day of that year. "
		"For example, if the CV says '2020', use '2020-01-01'. "
		"If the CV says 'Present', 'Current', or an equivalent term for an "
		"employment end date, set currently_employed to true and leave end_date empty. "
		"Do not calculate years_of_experience yourself; it will be calculated "
		"separately by the application. "
		"Do not infer missing employment dates from surrounding information. "
		"Set source_information.source_file to the supplied file name and "
		"do not include commentary outside the structured result."
        ),
    )

    # Run extraction
    result = agent.run_sync(
        f"Source file name: {path.name}\n\n"
        f"CV text:\n{extracted_text}"
    )

    cv = result.output

    # Calculate years of experience from the extracted employment dates.
    cv.professional_profile.years_of_experience = (
        calculate_years_of_experience(cv.work_experience)
    )

    # Write JSON
    Path(output_path).write_text(
        cv.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return cv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PDF CV to structured JSON."
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the CV PDF",
    )

    parser.add_argument(
        "output_path",
        type=Path,
        help="Path where the JSON file should be written",
    )

    parser.add_argument(
        "--api-key",
        help="Google API key; defaults to GOOGLE_API_KEY",
    )

    args = parser.parse_args()

    convert_cv_to_json(
        args.pdf_path,
        args.output_path,
        args.api_key,
    )


if __name__ == "__main__":
    main()
