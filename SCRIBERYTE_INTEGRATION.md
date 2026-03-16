# Scriberyte ↔ WoundCare AI — Integration Guide

> **Last Updated:** 2026-02-19

---

## How It Works

```
Scriberyte uploads audio to S3
        │
        ▼
Step 1 → Transcribe (OpenAI Whisper)
Step 2 → Fetch patient info from Scriberyte DB (MeetingId lookup)
Step 3 → LLM parse transcript → structured wound data (Gemini 3 Pro)
Step 4 → Generate HTML chart → save to S3
```

---

## Scriberyte Responsibilities

### 1. Platform Setup

- Create a **Wound Care** specialty
- Configure audio recording for wound care providers
- Map provider(s) to the specialty

### 2. Upload Audio to S3

Upload recordings to:

```
s3://woundcare-bucket/woundcare/{provider_uuid}/split-audio-files/{meeting_id}.wav
```

**`meeting_id`** = UUID from the `ZoomMeeting` table.

| Type | Filename | Example |
|------|----------|---------|
| New visit | `{meeting_id}.wav` | `fff73389-4bd0-40e5-8cb5-54411b1fadba.wav` |
| Addendum | `{meeting_id}_add.wav` | `fff73389-4bd0-40e5-8cb5-54411b1fadba_add.wav` |

- `_add` suffix → patches existing chart (does not create a new one)
- Supported formats: `.wav`, `.mp3`, `.m4a`, `.ogg`

**Upload code:**
```python
import boto3

s3 = boto3.client("s3",
    aws_access_key_id="ACCESS_KEY",
    aws_secret_access_key="SECRET_KEY",
    region_name="us-east-1"
)
s3.upload_file("recording.wav", "woundcare-bucket",
    "woundcare/FBD80A28-D6A6-4A8C-B7BB-5C478913AB5D/split-audio-files/fff73389-4bd0-40e5-8cb5-54411b1fadba.wav")
```

S3 credentials are stored in AWS Secrets Manager:
```bash
aws secretsmanager get-secret-value \
  --secret-id woundcare/scriberyte-s3-credentials \
  --query SecretString --output text
```

> **Write-only** — Scriberyte can only upload to `woundcare/audio/`. Cannot read charts or state.

---

## Patient Info — DB Query (Already Configured)

We query the Scriberyte MSSQL DB directly using `MeetingId`:

```sql
SELECT
    P.FirstName, P.LastName, P.Age, P.Gender,
    P.DateOfService, PA.AppointmentEnd,
    EPD.DOB, EPD.Age,
    PR.Name AS PhysicianName,
    H.Name  AS FacilityName
FROM Patient AS P
INNER JOIN PatientAppointment AS PA ON PA.PatientId = P.ID
LEFT JOIN EMRPatientDetails EPD
    ON CAST(EPD.MRN AS NVARCHAR(50)) = CAST(P.MRN AS NVARCHAR(50))
INNER JOIN ZoomMeeting AS ZM ON ZM.PatientAppointmentId = PA.ID
LEFT JOIN Provider AS PR ON PR.Id = ZM.physicianId
LEFT JOIN Hospital AS H  ON H.Id = ZM.facilityId
WHERE ZM.MeetingId = :meeting_id
    AND PA.IsActive = 1
    AND P.IsActive = 1;
```

**Fields extracted:**

| Field | Source | Maps To |
|-------|--------|---------|
| FirstName + LastName | Patient | Patient Name |
| DOB | EMRPatientDetails | Date of Birth |
| DateOfService / AppointmentEnd | Patient / PatientAppointment | Visit Date |
| Age, Gender | Patient | Demographics (context sentence for LLM) |
| PhysicianName | Provider | Physician on chart |
| FacilityName | Hospital | Facility on chart |

> No API needed — direct DB connection. If credentials change, update Secrets Manager (`woundcare/scriberyte-db-credentials`).

---

## Output — What We Deliver

After processing, charts are at:

```
s3://woundcare-bucket/woundcare/{provider_id}/chatgpt_htmls/{meeting_id}/latest.html
s3://woundcare-bucket/woundcare/{provider_id}/chatgpt_htmls/{meeting_id}/history/v1.html
```

Addendums increment the version (`v2.html`, `v3.html`, ...) and overwrite `latest.html`.

---

## Checklist (Scriberyte Team)

- [ ] Create Wound Care specialty on platform
- [ ] Configure audio recording for providers
- [ ] Confirm `MeetingId` format matches `ZoomMeeting.MeetingId`
- [ ] Implement S3 upload on recording completion
- [ ] Test with a sample audio file
- [ ] Confirm chart retrieval method (S3 direct or API)
