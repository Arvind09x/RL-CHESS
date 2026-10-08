"""Record exact locally downloaded bytes and the actual curl tool observations."""
from datetime import datetime,timezone
import hashlib,json,zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity
root=Path(__file__).resolve().parents[2]
archive=Path('/private/tmp/lichess_elite_2025-01.zip')
manifest=DatasetManifest.load(root/'data/manifests/elite-2025-01-v1.json')
identity=file_identity(archive)
assert (identity['sha256'],identity['bytes'])==(manifest.archive.downloaded_sha256,manifest.archive.downloaded_bytes)
retrieved=datetime.fromtimestamp(archive.stat().st_mtime,timezone.utc).isoformat()
assert retrieved==manifest.archive.retrieved_at_utc
headers=Path('/private/tmp/d02-download-headers.txt').read_bytes()
(root/'runs/d02/download-headers.json').write_text(json.dumps({'raw_headers_latin1':headers.decode('latin1'),'bytes':len(headers),'sha256':hashlib.sha256(headers).hexdigest()},indent=2,sort_keys=True)+'\n')
members=[]
with zipfile.ZipFile(archive) as zipped:
 for index,info in enumerate(zipped.infolist(),1):
  with zipped.open(info) as h:
   digest=hashlib.file_digest(h,'sha256').hexdigest()
  members.append(dict(member_index=index,name=info.filename,bytes=info.file_size,compressed_bytes=info.compress_size,crc32=f'{info.CRC:08x}',sha256=digest,compression=info.compress_type,zip_timestamp=list(info.date_time),header_offset=info.header_offset))
argv=['curl','--fail','--location','--max-time','240','--dump-header','/private/tmp/d02-download-headers.txt','--write-out','%{json}',manifest.archive.url,'--output',str(archive)]
record={'archive_path':str(archive),'requested_url':manifest.archive.url,'final_url':manifest.archive.url,'redirect_count':0,'redirects':[],'http_code':200,'downloaded_sha256':identity['sha256'],'downloaded_bytes':identity['bytes'],'retrieved_at_utc':retrieved,'retrieval_time_basis':'download output file mtime in UTC, observed after curl completed; not server Last-Modified','headers_evidence':'runs/d02/download-headers.json','previous_index_head_evidence':'runs/d01/archive-provenance.json','members':members,'member_count':len(members),'pgn_member_count':sum(i['name'].lower().endswith('.pgn') for i in members),'curl_observations':{'exitcode':0,'http_code':200,'num_redirects':0,'url_effective':manifest.archive.url,'size_download':82724777,'time_total':7.792317,'ssl_verify_result':0,'content_type':'application/zip'},'observation_source':'actual functions.exec / exec_command curl response on 2026-10-08; selected fields transcribed, response headers retained verbatim','attempts':[{'argv':argv,'cwd':str(root),'sandbox':'use_default','exit_code':6,'stderr':'curl: (6) Could not resolve host: database.nikonoel.fr','downloaded_bytes':0},{'argv':argv,'cwd':str(root),'sandbox':'require_escalated (user approved)','exit_code':0,'downloaded_bytes':82724777}]}
(root/'runs/d02/archive-provenance.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True))
