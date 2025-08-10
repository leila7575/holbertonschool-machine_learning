#!/usr/bin/env python3
"""Displays first launch information"""


import requests

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v3/launches"
    response = requests.get(url)

    if response.status_code == 200:
        launch_data = response.json()
        launch_data.sort(key=lambda x: x.get('launch_date_unix', float('inf')))
        launch = launch_data[0]
        rocket_name = launch['rocket']['rocket_name']
        launchpad_name = launch['launch_site']['site_name']
        launchpad_locality = launch['launch_site']['site_name_long']
        print(
            f"{launch['mission_name']} ({launch['launch_date_local']}) "
            f"{rocket_name} - {launchpad_name} ({launchpad_locality})"
        )
    else:
        print('Not found')
