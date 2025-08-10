#!/usr/bin/env python3
"""Prints user locations"""


import sys
import requests
import time


def user_location(url):
    """Prints user locations"""
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(data.get('location'))
        elif response.status_code == 403:
            reset = int(response.headers.get('X-Ratelimit-Reset', 0))
            now = int(time.time())
            X = (reset - now) // 60
            print(f"Reset in {X} min")
        else:
            print("Not found")

    except requests.exceptions.RequestException:
        print("Not found")


if __name__ == '__main__':

    url = sys.argv[1]
    user_location(url)
