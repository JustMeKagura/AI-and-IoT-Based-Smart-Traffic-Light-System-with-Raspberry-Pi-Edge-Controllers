ATCS/
├── config.yaml                  # Single source of truth (Broker IP, thresholds, timing defaults)
├── requirements.txt             # Global dev deps (pytest, black, ruff, pyyaml, etc.)
├── README.md
├── .gitignore
│
├── shared/                      # Contracts & Config Loader (imported by Edge & Server)
│   ├── config_loader.py         # Parses config.yaml + env vars, validates types
│   ├── schemas.py               # FramePayload, PhaseDecision dataclasses
│   ├── constants.py             # Phase IDs, safety limits, COCO classes [2,3,5,7]
│   └── version.py               # Dev sync check
│
├── edge/                        # 🥧 Raspberry Pi 5 ("The Body")
│   ├── main.py                  # Orchestrator: spawns Reporter & Controller threads
│   ├── requirements_edge.txt    # opencv-python-headless, paho-mqtt, RPi.GPIO, PyYAML
│   │
│   ├── hardware/
│   │   ├── camera.py            # V4L2 capture, resize 640px, JPEG encode
│   │   └── gpio_driver.py       # GPIO pin control, relay/LED interface
│   │
│   ├── logic/                   # Local safety & state validation
│   │   ├── state_machine.py     # Red→Green→Amber transitions, rejects unsafe commands
│   │   ├── safety.py            # Watchdog, cold start (All Red), 30s timeout fallback
│   │   ── timer.py             # Precise phase timing loops (non-blocking)
│   │
│   └── network/
│       └── mqtt_client.py       # Paho wrapper: publishes frames, subscribes to commands
│
├── server/                      # 🖥️ Laptop/PC ("The Brain")
│   ├── main.py                  # Orchestrator: inference loop + decision publisher
│   ├── requirements_server.txt  # ultralytics, opencv-python, paho-mqtt, numpy, pyyaml
│   │
│   ├── inference/               # Vision pipeline
│   │   ├── preprocessor.py      # Base64→BGR, CLAHE, resize, denoise
│   │   ├── detector.py          # YOLOv8s inference (conf=0.15, imgsz=640)
│   │   └── counter.py           # Filters COCO IDs, aggregates vehicles per lane
│   │
│   ├── logic/                   # Decision engine
│   │   ├── timing_algo.py       # Dynamic phase duration math (Webster/adaptive)
│   │   └── smoother.py          # Moving average (last 3 frames) to prevent flicker
│   │
│   ├── persistence/
│   │   └── database.py          # SQLite3 thread-safe logger (events, counts, decisions)
│   │
│   └── network/
│       └── mqtt_client.py       # Paho wrapper: subscribes to frames, publishes decisions
