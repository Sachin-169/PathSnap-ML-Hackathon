# Folium route map

`generate_folium_map.py` writes a standalone interactive HTML map for a student-test trip. It loads the same model and label-free trip registry used by FastAPI, performs candidate classification and HMM/Viterbi decoding, then displays the road network, GPS observations, predicted route, and start/end markers with layer controls.

```bash
python visualizations/generate_folium_map.py 00000009
```

By default the file is saved to `artifacts/maps/trip_00000009.html`. Optional comparison layer:

```bash
python visualizations/generate_folium_map.py 00000009 \
  --reference-zip "hackathon_materials/Student folder.zip"
```

The reference `.route` is only added as an initially hidden visualization after inference. It never enters the model features, classifier, or route decoder. Folium's OpenStreetMap base layer requires internet access and keeps OpenStreetMap attribution visible.
