# FastAPI Frontend Skeleton

## What this includes
- Browser UI for editing `input_parameters/variables.json` and `input_parameters/global_parameters.json`
- Single-variable editor with dropdown selection
- Add/delete variables
- Add/remove band rows for the selected variable
- Include/exclude toggles per variable for data creation
- Factory reset for variables via `input_parameters/variables.factory.json`
- API routes to validate and save configuration (with `.bak` backup files)
- API routes to run pipeline stages asynchronously and stream job logs

## Start the app
From the project root:

```powershell
python -m uvicorn frontend.main:app --reload
```

Then open: `http://127.0.0.1:8000`

## API endpoints
- `GET /api/config`
- `POST /api/config/validate`
- `PUT /api/config`
- `POST /api/config/variables/factory-reset`
- `POST /api/pipeline/jobs`
- `GET /api/pipeline/jobs`
- `GET /api/pipeline/jobs/{job_id}`
