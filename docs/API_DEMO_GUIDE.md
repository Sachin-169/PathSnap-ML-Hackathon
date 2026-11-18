# PathSnap API demo guide

This guide explains how to run the evaluator interface across the two laptops
used in the PathSnap demonstration.

## Laptop 1: model and API

Laptop 1 holds the project, trip files, and trained model. Open PowerShell in
the repository root and start the server:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --access-log
```

Keep this terminal open during the demo. It prints each API request and a
summary of each prediction, including the trip ID, result status, number of
GPS points, number of predicted road arcs, and any route decoder notes.

To find Laptop 1's local network address, open a second PowerShell terminal
and run:

```powershell
ipconfig
```

Use the IPv4 address for the active Wi-Fi adapter.

## Laptop 2: evaluator interface

Connect Laptop 2 to the same Wi-Fi network. Open a browser and enter:

```text
http://<laptop-1-ip>:8000
```

Replace `<laptop-1-ip>` with the IPv4 address shown on Laptop 1, such as
`192.168.1.5`. Do not use `0.0.0.0` in the browser; that address tells the
server which network interfaces to listen on.

The UI lists the trips available under `dataset/student_test/`. Select a trip
and choose **Predict route**. The GPS trace and predicted road sequence appear
in the browser on Laptop 2. The corresponding request and any route notes
appear in Laptop 1's server terminal.

## API endpoints

- `GET /api/v1/health` reports whether the API and prediction pipeline are
  ready, and how many student-test trips are available.
- `GET /api/v1/trips` lists complete trips in `dataset/student_test/`.
- `POST /api/v1/predict` accepts a trip ID and returns the decoded route.
- `GET /docs` opens FastAPI's interactive endpoint documentation.

The prediction loader reads `.track`, `.nodes`, and `.arcs`. It does not read
the trip's `.route` reference labels during inference. A partial route status
means the decoder found one or more gaps or infeasible transitions; the UI
shows the notes and keeps those route sections separate.

## If Laptop 2 cannot connect

First open `http://127.0.0.1:8000` on Laptop 1. If that works, check that both
laptops are on the same network, that Laptop 2 is using Laptop 1's current
IPv4 address, and that the Windows firewall permits the Python server on the
private network.
