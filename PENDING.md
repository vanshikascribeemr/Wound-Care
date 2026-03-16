# WoundCare AI — Pending Tasks

> Last updated: 2026-02-27

---

## 🔴 P1 — Deploy Lambda Architecture

### ✅ Dev Team — Already Done
- [x] Lambda handlers written (`src/lambda_*.py`)
- [x] CDK stack written (`infra/stacks/woundcare_stack.py`)
- [x] Deployment guide written (`CLOUD_ENGINEERING.md`)
- [x] All tests passing (8/8)

### 🔲 Cloud Engineer — To Do
> Hand off: **repo + `CLOUD_ENGINEERING.md`** — they can deploy independently

- [ ] Build Lambda layer (needs Linux — see `CLOUD_ENGINEERING.md` Step 1)
- [ ] Run `cdk bootstrap aws://ACCOUNT_ID/us-east-1`
- [ ] Run `cdk deploy`
- [ ] Set Google API key in Secrets Manager
- [ ] Set OpenAI API key in Secrets Manager
- [ ] Set Scriberyte DB credentials in Secrets Manager (`woundcare/scriberyte-db-credentials`)
- [ ] Share Scriberyte S3 credentials with Scriberyte team
- [ ] Set GitHub secrets for CI/CD
- [ ] End-to-end test: upload audio → verify chart generated

### 🔲 Scriberyte Team — To Do
> Request from us to Scriberyte platform team

- [ ] Create **Wound Care** specialty on Scriberyte platform
- [ ] Configure audio recording flow for wound care providers
- [ ] Map wound care provider(s) to the new specialty
- [ ] Confirm `MeetingId` format used when audio is saved to S3
- [ ] Share test provider credentials for end-to-end testing

---

## ✅ ~~P1.5 — S3 Folder Organization~~ DONE
**Completed 2026-02-27** — Reorganized S3 into provider-centric folders.

- [x] Implemented `{prefix}/{provider_uuid}/` structure.
- [x] Folders: `split-audio-files/`, `audio/`, `transcribed-speaker-label/`, `clinical_data_jsons/`, `chatgpt_htmls/`.
- [x] Implemented traceable custom naming (e.g., `-mp3-prod01-wisper-...`) for automated audits.
- [x] Switched from DOCX to professional responsive HTML charts.
- [x] Updated all documentation to reflect the new architecture.

---

## 🔴 P2 — Scribe Evaluation Flow

**Task:** Build the chart review + approval workflow for scribes

**Open questions (need answers before building):**
- [ ] How does scribe access the chart? (S3 direct / Scriberyte dashboard / custom UI?)
- [ ] How does scribe send corrections? (Addendum audio / typed edits / approval button?)
- [ ] Where does approved chart go? (Scriberyte EMR / S3 only / both?)

**Once clarified, build:**
- [ ] Notification when chart is ready (email / webhook to Scriberyte)
- [ ] Scribe correction endpoint (`POST /corrections/{appointment_id}`)
- [ ] Approval endpoint (`POST /approve/{appointment_id}`)
- [ ] Track correction history in JSON state
- [ ] Final delivery to EMR on approval

---

## ✅ ~~P3 — Scriberyte Patient Info Integration~~ DONE

**Completed 2026-02-19** — Switched from REST API to direct MSSQL DB.

- [x] DB connection configured (MSSQL → `ScribeRyteQA`)
- [x] SQL query: `ZoomMeeting → PatientAppointment → Patient + EMRPatientDetails`
- [x] `scriberyte_client.py` rewritten (API → DB via SQLAlchemy + pyodbc)
- [x] `lambda_patient_info.py` updated
- [x] `manager.py` updated (sync call, context sentence prepended to transcript)
- [x] ODBC driver auto-detection (SQL Server / Driver 17 / Driver 18)
- [x] MRN type mismatch fixed (CAST to NVARCHAR)
- [x] DB connection tested successfully

---

## 🟡 P4 — Cost Tracker

**Task:** Add automated cost tracking and budget alerts

**What to build:**
- [ ] AWS Budgets alert in CDK — email when monthly bill > $100
- [ ] CloudWatch dashboard in CDK — visits/day, latency, errors
- [ ] Per-visit cost logging in `lambda_generate.py`
  - Log audio duration, tokens used, estimated cost per visit
  - Save to `woundcare/{provider_id}/costs/{appointment_id}.json`

---

## 🟢 P5 — Accuracy Scoring (Nice to Have)

**Task:** Add confidence scoring to LLM output + highlight low-confidence fields in HTML

**What to build:**
- [ ] Update `INTENT_EXTRACTION_PROMPT` to request confidence score per field
- [ ] Update `EncounterState` model to store confidence scores
- [ ] Highlight low-confidence fields (< 80%) in HTML chart for scribe attention
- [ ] Log accuracy metrics per visit for trend analysis

---

## ✅ Completed

| Date | Task |
|------|------|
| 2026-02-27 | Multi-provider S3 structure implementation (UUID folders) |
| 2026-02-27 | Transition from DOCX to HTML chart generation |
| 2026-02-19 | Scriberyte integration: REST API → MSSQL DB (direct patient lookup) |
| 2026-02-19 | DB connection tested & working (`ScribeRyteQA` @ 3.140.148.166) |
| 2026-02-19 | Lambda + Step Functions architecture implemented |
| 2026-02-18 | Gemini API unblocked — `gemini-3-pro-preview` working |
| 2026-02-18 | CI/CD pipeline — GitHub Actions (Test → CDK Deploy) |
