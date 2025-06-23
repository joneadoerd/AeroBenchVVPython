import json

example = {
    "targets": [
        {
            "id": 1,
            "init_state": {
                "vt": 500.0,
                "alpha": 0.05,
                "beta": 0.01,
                "phi": 0.0,
                "theta": 0.0,
                "psi": 0.0,
                "p": 0.0,
                "q": 0.0,
                "r": 0.0,
                "pn": 1000.0,
                "pe": 2000.0,
                "h": 10000.0,
                "pow": 5.0
            },
            "waypoints": [
                {"alt": 10000.0, "lat": 1000.0, "lon": 2000.0},
                {"alt": 12000.0, "lat": 2000.0, "lon": 3000.0}
            ]
        },
        {
            "id": 2,
            "init_state": {
                "vt": 520.0,
                "alpha": 0.04,
                "beta": 0.02,
                "phi": 0.1,
                "theta": 0.0,
                "psi": 0.2,
                "p": 0.0,
                "q": 0.0,
                "r": 0.0,
                "pn": 1500.0,
                "pe": 2500.0,
                "h": 11000.0,
                "pow": 6.0
            },
            "waypoints": [
                {"alt": 11000.0, "lat": 1500.0, "lon": 2500.0},
                {"alt": 13000.0, "lat": 2500.0, "lon": 3500.0}
            ]
        }
    ],
    "time_step": 0.1,
    "max_time": 20.0
}

print(json.dumps(example, indent=2))
