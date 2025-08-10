#!/usr/bin/env python3
"""availableShips returns list of ships from Swapi API"""


import requests


def availableShips(passengerCount):
    """Returns list of ships given a number of passengers"""
    url = 'https://swapi-api.hbtn.io/api/starships/'
    ships = []
    while url:
        r = requests.get(url)
        data = r.json()
        for starship in data["results"]:
            passengers = starship['passengers']

            try:
                passengers_int = int(passengers)
            except ValueError:
                continue

            if passengers_int >= passengerCount:
                ships.append(starship['name'])

        url = data.get("next")
    return (ships)
