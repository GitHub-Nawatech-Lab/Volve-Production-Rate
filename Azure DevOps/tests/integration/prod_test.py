import requests
import json

test_sample = json.dumps({'data': [[149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0], [149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0, 149.86320688, 48.82845551, 145.716024, 7.68319231, 17.05146447, 0, 0]]})
test_sample = str(test_sample)

def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, test_sample, headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0