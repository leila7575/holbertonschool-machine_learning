#!/usr/bin/env python3
"""Displays the number of launches per rocket"""


import requests


def number_launches():
    """displays the number of launches per rocket"""
    try:
        launches_response = requests.get(
            'https://api.spacexdata.com/v4/launches'
        )
        launches_response.raise_for_status()
        launches_data = launches_response.json()

        rockets_response = requests.get(
            'https://api.spacexdata.com/v4/rockets'
        )
        rockets_response.raise_for_status()
        rockets_data = rockets_response.json()

        rocket_dict = {rocket['id']: rocket['name'] for rocket in rockets_data}

        count = {}
        for launch in launches_data:
            rocket_id = launch['rocket']
            if rocket_id and rocket_id in rocket_dict:
                rocket_name = rocket_dict[rocket_id]
                count[rocket_name] = count.get(rocket_name, 0) + 1

        sorted_count = sorted(count.items(), key=lambda x: (-x[1], x[0]))

        for rocket, value in sorted_count:
            print(f"{rocket}: {value}")
    except Exception:
        print("Error")


if __name__ == '__main__':
    number_launches()
