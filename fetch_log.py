import os, json, urllib.request, base64

config_dir = os.environ.get('KAGGLE_CONFIG_DIR', os.path.expanduser('~/.kaggle'))
with open(os.path.join(config_dir, 'kaggle.json')) as f:
    creds = json.load(f)

username = creds['username']
key = creds['key']
auth = base64.b64encode(f'{username}:{key}'.encode()).decode()

url = 'https://www.kaggle.com/api/v1/kernels/output?userName=ladyfaye&kernelSlug=triagegeist-triage-acuity-prediction'
req = urllib.request.Request(url)
req.add_header('Authorization', f'Basic {auth}')

resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode('utf-8'))
log = data.get('log', '')

out_path = 'c:/Users/feika/Documents/Projects/Competitions/triagegeist/outputs/klog.txt'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(log)
print(f'Log saved: {len(log)} chars')
