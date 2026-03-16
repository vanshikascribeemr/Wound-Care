"""
WoundCare AI - Scriberyte DB Client (src/scriberyte_client.py)
----------------------------------------------------------------
Fetches patient appointment details from the Scriberyte database (MSSQL)
before chart generation, so patient demographics are pre-populated
without requiring the provider to dictate them.

Flow:
  1. Watcher detects new audio file with meeting_id (appointment_id)
  2. EncounterManager calls ScriberyteClient.fetch_patient_info(meeting_id)
  3. Returns PatientInformation + a context sentence for the LLM
  4. Pre-populated into EncounterState before LLM parsing

Configuration (in .env):
  SCRIBERYTE_DB_SERVER   - MSSQL server hostname
  SCRIBERYTE_DB_NAME     - Database name
  SCRIBERYTE_DB_USERNAME - Database username
  SCRIBERYTE_DB_PASSWORD - Database password
  SCRIBERYTE_SOURCE_ID   - Source ID filter (optional)

Graceful Fallback:
  If not configured or DB call fails, the system continues without patient info
  and relies on the transcript for any available demographics.
"""
import os
import logging
import pyodbc
from typing import Optional, Tuple
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .models import PatientInformation

logger = logging.getLogger(__name__)


def _execute_fetchone(session_factory, sql_query: str, params: dict):
    """Execute a SQL query and return the first row."""
    session = session_factory()
    try:
        result = session.execute(text(sql_query), params)
        return result.fetchone()
    except Exception as e:
        logger.error(f"DB query failed: {e}")
        raise
    finally:
        session.close()


class ScriberyteClient:
    """
    Fetches patient details from Scriberyte's MSSQL database.
    Called by the pipeline before chart generation to pre-populate patient info.
    """

    def __init__(self):
        self.db_server = os.getenv("SCRIBERYTE_DB_SERVER", "")
        self.db_name = os.getenv("SCRIBERYTE_DB_NAME", "")
        self.db_username = os.getenv("SCRIBERYTE_DB_USERNAME", "")
        self.db_password = os.getenv("SCRIBERYTE_DB_PASSWORD", "")
        self.source_id = os.getenv("SCRIBERYTE_SOURCE_ID", "")

        self._engine = None
        self._Session = None

    def is_configured(self) -> bool:
        """Check if Scriberyte DB integration is configured."""
        return bool(self.db_server and self.db_name and self.db_username and self.db_password)

    def _get_session(self):
        """Lazy-initialize the SQLAlchemy engine and session factory."""
        if self._Session is None:
            # Auto-detect best available ODBC driver
            available = pyodbc.drivers()
            if 'ODBC Driver 18 for SQL Server' in available:
                driver = 'ODBC+Driver+18+for+SQL+Server'
                extras = '&Encrypt=yes&TrustServerCertificate=yes'
            elif 'ODBC Driver 17 for SQL Server' in available:
                driver = 'ODBC+Driver+17+for+SQL+Server'
                extras = '&Encrypt=yes&TrustServerCertificate=yes'
            else:
                driver = 'SQL+Server'
                extras = ''
            
            sql_conn_str = (
                f"mssql+pyodbc://{self.db_username}:{quote_plus(self.db_password)}"
                f"@{self.db_server}/{self.db_name}"
                f"?driver={driver}{extras}"
            )
            self._engine = create_engine(
                sql_conn_str,
                pool_size=10,
                max_overflow=20,
                pool_recycle=360,
                pool_timeout=60,
            )
            self._Session = sessionmaker(bind=self._engine)
        return self._Session

    def fetch_patient_info(self, meeting_id: str) -> Tuple[Optional[PatientInformation], str]:
        """
        Fetch patient details from Scriberyte DB using meeting_id.

        Returns:
            Tuple of (PatientInformation or None, context_sentence: str)
            The context_sentence is a clinical-style sentence prepended to the
            transcript to give the LLM demographic context.
        """
        if not self.is_configured():
            logger.info("Scriberyte DB not configured. Skipping patient info fetch.")
            return None, ""

        try:
            Session = self._get_session()

            sql_query = """
                SELECT
                    P.FirstName,
                    P.LastName,
                    P.Age,
                    P.Gender,
                    P.DateOfService,
                    PA.AppointmentEnd,
                    EPD.DOB,
                    EPD.Age,
                    PR.Name AS PhysicianName,
                    H.Name AS FacilityName,
                    PA.ReasonOfVisit,
                    PVT.VisitType,
                    PA.AppointmentOn
                FROM Patient AS P
                INNER JOIN PatientAppointment AS PA
                    ON PA.PatientId = P.ID
                LEFT JOIN EMRPatientDetails EPD
                    ON CAST(EPD.MRN AS NVARCHAR(50)) = CAST(P.MRN AS NVARCHAR(50))
                INNER JOIN ZoomMeeting AS ZM
                    ON ZM.PatientAppointmentId = PA.ID
                LEFT JOIN Provider AS PR
                    ON PR.Id = ZM.physicianId
                LEFT JOIN Hospital AS H
                    ON H.Id = ZM.facilityId
                LEFT JOIN PatientVisitTypes AS PVT
                    ON PVT.Id = PA.VisitTypeId
                WHERE ZM.MeetingId = :meeting_id
                    AND PA.IsActive = 1
                    AND P.IsActive = 1;
            """
            result = _execute_fetchone(Session, sql_query, {'meeting_id': meeting_id})

            # Set defaults
            FirstName = ""
            LastName = ""
            Age = ""
            Gender = ""
            DateOfService = ""
            AppointmentEnd = ""
            EMR_DOB = ""
            EMR_Age = ""
            PhysicianName = ""
            FacilityName = ""
            ReasonOfVisit = ""
            VisitType = ""
            AppointmentStart = ""

            if result:
                FirstName = result[0] if result[0] else ""
                LastName = result[1] if result[1] else ""
                Age = result[2] if result[2] else ""
                Gender = result[3] if result[3] else ""
                DateOfService = str(result[4]) if result[4] else ""
                AppointmentEnd = str(result[5]) if result[5] else ""
                EMR_DOB = result[6] if result[6] else ""
                EMR_Age = result[7] if result[7] else ""
                PhysicianName = result[8] if result[8] else ""
                FacilityName = result[9] if result[9] else ""
                ReasonOfVisit = result[10] if result[10] else ""
                VisitType = result[11] if result[11] else ""
                AppointmentStart = str(result[12]) if result[12] else ""

            logger.info(f"FirstName: {FirstName}, LastName: {LastName}, Age: {Age}, Gender: {Gender}")
            logger.info(f"DateOfService: {DateOfService}, EMR_DOB: {EMR_DOB}, EMR_Age: {EMR_Age}")
            logger.info(f"ReasonOfVisit: {ReasonOfVisit}, VisitType: {VisitType}, AppointmentStart: {AppointmentStart}")

            # Derive composite fields
            PatientName = f"{FirstName} {LastName}".strip() if FirstName or LastName else ""
            
            # Use full datetime: prefer AppointmentStart (has time), fallback to DateOfService
            if AppointmentStart:
                VisitDateTime = AppointmentStart
            elif DateOfService:
                VisitDateTime = DateOfService
            else:
                VisitDateTime = AppointmentEnd

            if EMR_Age:
                Age = EMR_Age

            if Gender in ['NA', 'Other', 'O']:
                Gender = None

            # ── Build clinical context sentence ────────────────────────
            parts = []

            # 1. Subject Identification
            if PatientName:
                parts.append(f"The patient, identified as {PatientName}")
            else:
                parts.append("The patient")

            # 2. Demographics (Age and Gender)
            if Age and Gender:
                parts.append(f"is a {Age}-year-old {Gender}")
            elif Age:
                parts.append(f"is a {Age}-year-old individual")
            elif Gender:
                parts.append(f"is a {Gender}")

            # 3. Date of Birth (DOB)
            if EMR_DOB:
                if Age or Gender:
                    parts.append(f"with a documented DOB of {EMR_DOB}")
                else:
                    parts.append(f"has a documented date of birth of {EMR_DOB}")

            # 4. Visit Date (Encounter Information)
            if VisitDateTime:
                parts.append(f"and was seen for the current encounter on {VisitDateTime}")

            # Join and ensure ending period
            sentence = " ".join(parts).strip()
            if not sentence.endswith("."):
                sentence += "."

            logger.info(f"Patient context sentence: {sentence}")

            # ── Build PatientInformation object ────────────────────────
            patient_info = PatientInformation(
                patient_name=PatientName or None,
                dob=str(EMR_DOB) if EMR_DOB else None,
                date_of_service=VisitDateTime or None,
                physician=PhysicianName or None,
                facility=FacilityName or None
            )

            return patient_info, sentence

        except Exception as e:
            logger.error(f"Scriberyte DB call failed: {e}. Will use transcript data only.")
            return None, ""
