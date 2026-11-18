# PathSnap evaluator UI handoff scaffold

Open `index.html` in a browser on Laptop 2. Click **Load trips**, choose one
of the trips returned by Laptop 1, then click **Show route**. The UI displays
the trip road network, GPS observations, predicted road arcs, route arc IDs,
and decoder diagnostics. Use the map layer checkboxes to show or hide each
layer. Expand the data panels below the map to inspect the GPS and road rows.

When Laptop 2 and Laptop 1 are separate computers, set **Laptop 1 API address**
to `http://<LAPTOP-1-WIFI-IP>:8000`. Do not leave it as `localhost`: on Laptop
2, that would point back to Laptop 2 itself.

For a local static server, run this on Laptop 2 from the project root:

```bash
cd ui
python3 -m http.server 8080 --bind 0.0.0.0
```

Then open `http://localhost:8080` on Laptop 2. The UI uses Leaflet and the
OpenStreetMap standard tile service for the colored geographic basemap. An
internet connection is required. It requests tiles only for the map currently
being viewed and shows the required OpenStreetMap attribution. The team still
needs the required Folium map output for the technical report.
