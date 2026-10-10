import cv2
import pytest
import requests

url = "http://127.0.0.1:2137/gaze_vector"
frame = cv2.imread(r"Alanwilder.png")
no_face_frame = cv2.imread(r"boxes.jpg")
response = requests.post(url, json={"frame": frame.tolist()})

def test_api_response__check_response_status_code__200():
    assert response.status_code==200

def test_api_response__if_data_is_int__yes():
    data = response.json()
    assert type(data['x'])==int
    assert type(data['y']) == int

def test_api_response__are_in_screen_bounds__yes():
    data = response.json()
    assert 0 <= data["x"] <= 1920
    assert 0 <= data["y"] <= 1080

def test_api_response__are_null__no():
    data = response.json()
    assert data["x"]!=None
    assert data["y"]!=None

def test_api_response__wrong_input_type__fail():
    fake_response = requests.post(url, json={"frame": 12}).json()
    assert fake_response["error"]=="not list"

def test_api_response__wrong_input_size__fail():
    fake_response = requests.post(url, json={"frame": [1,2,3,4]}).json()
    assert fake_response["error"]

def test_api_response__no_face__fail():
    fake_response = requests.post(url, json={"frame": no_face_frame.tolist()}).json()
    print(fake_response)
    assert fake_response["error"]=="Couldn't find face points"