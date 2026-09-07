"""Convert a PDF CV into the application's structured JSON format."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pypdf import PdfReader


class Location(BaseModel):
	city: str = ""
	region: str = ""
	country: str = ""
	postal_code: str = ""


class PersonalInformation(BaseModel):
	full_name: str = ""
	first_name: str = ""
	last_name: str = ""
	email: str = ""
	phone: str = ""
	location: Location = Field(default_factory=Location)
	linkedin_url: str = ""
	portfolio_url: str = ""
	other_urls: list[str] = Field(default_factory=list)


class ProfessionalProfile(BaseModel):
	headline: str = ""
	summary: str = ""
	years_of_experience: float | None = None
	current_job_title: str = ""
	current_industry: str = ""


class WorkExperience(BaseModel):
	job_title: str = ""
	company: str = ""
	location: str = ""
	employment_type: str = ""
	start_date: str = ""
	end_date: str = ""
	currently_employed: bool = False
	description: str = ""
	responsibilities: list[str] = Field(default_factory=list)
	achievements: list[str] = Field(default_factory=list)
	skills_used: list[str] = Field(default_factory=list)
	industry: str = ""


class Education(BaseModel):
	degree: str = ""
	field_of_study: str = ""
	institution: str = ""
	location: str = ""
	start_date: str = ""
	end_date: str = ""
	graduation_date: str = ""
	grade: str = ""
	description: list[str] = Field(default_factory=list)


class Skills(BaseModel):
	technical: list[str] = Field(default_factory=list)
	software: list[str] = Field(default_factory=list)
	programming_languages: list[str] = Field(default_factory=list)
	frameworks: list[str] = Field(default_factory=list)
	databases: list[str] = Field(default_factory=list)
	tools: list[str] = Field(default_factory=list)
	business: list[str] = Field(default_factory=list)
	soft_skills: list[str] = Field(default_factory=list)
	other: list[str] = Field(default_factory=list)


class Language(BaseModel):
	language: str = ""
	proficiency: str = ""


class Certification(BaseModel):
	name: str = ""
	issuing_organization: str = ""
	issue_date: str = ""
	expiry_date: str = ""
	credential_id: str = ""


class Project(BaseModel):
	name: str = ""
	description: str = ""
	role: str = ""
	technologies: list[str] = Field(default_factory=list)
	start_date: str = ""
	end_date: str = ""
	url: str = ""


class Volunteering(BaseModel):
	organization: str = ""
	role: str = ""
	start_date: str = ""
	end_date: str = ""
	description: list[str] = Field(default_factory=list)


class AdditionalInformation(BaseModel):
	driving_license: bool = False
	work_authorization: list[str] = Field(default_factory=list)
	security_clearance: str = ""
	availability: str = ""
	other: list[str] = Field(default_factory=list)


class SourceInformation(BaseModel):
	source_file: str = ""
	extracted_date: str = ""
	extraction_confidence: float | None = None


class CV(BaseModel):
	"""The JSON contract produced from a CV."""

	model_config = ConfigDict(extra="ignore")

	personal_information: PersonalInformation = Field(default_factory=PersonalInformation)
	professional_profile: ProfessionalProfile = Field(default_factory=ProfessionalProfile)
	work_experience: list[WorkExperience] = Field(default_factory=list)
	education: list[Education] = Field(default_factory=list)
	skills: Skills = Field(default_factory=Skills)
	languages: list[Language] = Field(default_factory=list)
	certifications: list[Certification] = Field(default_factory=list)
	projects: list[Project] = Field(default_factory=list)
	achievements: list[str] = Field(default_factory=list)
	publications: list[str] = Field(default_factory=list)
	volunteering: list[Volunteering] = Field(default_factory=list)
	additional_information: AdditionalInformation = Field(default_factory=AdditionalInformation)
	source_information: SourceInformation = Field(default_factory=SourceInformation)


def extract_pdf_text(pdf_path: str | Path) -> str:
	"""Extract text from every page of a PDF, preserving page boundaries."""
	path = Path(pdf_path)
	if path.suffix.lower() != ".pdf":
		raise ValueError("The CV file must be a PDF.")
	if not path.is_file():
		raise FileNotFoundError(f"CV file not found: {path}")

	reader = PdfReader(str(path))
	pages = [page.extract_text() or "" for page in reader.pages]
	text = "\n\n".join(pages).strip()
	if not text:
		raise ValueError("No text could be extracted from the PDF.")
	return text


def convert_cv_to_json(
	pdf_path: str | Path,
	output_path: str | Path,
	api_key: str | None = None,
) -> CV:
	"""Extract a PDF CV and write its AI-structured representation as JSON."""
	path = Path(pdf_path)
	extracted_text = extract_pdf_text(path)
	google_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
	if not google_api_key:
		raise ValueError("Provide api_key or set the GOOGLE_API_KEY environment variable.")

	provider = GoogleProvider(api_key=google_api_key)
	model = GoogleModel("gemini-flash-lite-latest", provider=provider)
	agent = Agent(
		model,
		output_type=CV,
		instructions=(
			"Extract facts from the CV text into the provided CV schema. "
			"Never invent information. Use empty strings, empty arrays, or null "
			"when a value is not present. Keep dates as written in the CV. "
			"Set source_information.source_file to the supplied file name and "
			"do not include commentary outside the structured result."
		),
	)
	result = agent.run_sync(
		f"Source file name: {path.name}\n\nCV text:\n{extracted_text}"
	)
	cv = result.output
	cv.source_information.source_file = path.name
	cv.source_information.extracted_date = (
		datetime.datetime.now(datetime.UTC).date().isoformat()
	)
	Path(output_path).write_text(cv.model_dump_json(indent=2), encoding="utf-8")
	return cv


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert a PDF CV to structured JSON.") 
	parser.add_argument("pdf_path", type=Path)
	parser.add_argument("output_path", type=Path)
	parser.add_argument("--api-key", help="Google API key; defaults to GOOGLE_API_KEY.")
	args = parser.parse_args()
	convert_cv_to_json(args.pdf_path, args.output_path, args.api_key)


if __name__ == "__main__":
	main()
