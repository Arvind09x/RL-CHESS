"""Independent raw header census of the entire selected member; not replay acceptance."""
import hashlib,json,zipfile
counts={'lines':0,'bytes':0,'event_headers':0,'setup_headers':0,'variant_headers':0,'brace_characters':0,'variation_characters':0}
with zipfile.ZipFile('/private/tmp/lichess_elite_2025-01.zip') as z:
 with z.open(z.infolist()[0]) as h:
  digest=hashlib.sha256()
  body_started=False
  for raw in h:
   digest.update(raw)
   counts['bytes']+=len(raw)
   counts['lines']+=1
   stripped=raw.strip()
   counts['event_headers']+=stripped.startswith(b'[Event ')
   counts['setup_headers']+=stripped.startswith(b'[SetUp ')
   counts['variant_headers']+=stripped.startswith(b'[Variant ')
   if stripped.startswith(b'['):
    body_started=False
   elif stripped:
    body_started=True
   if body_started:
    counts['brace_characters']+=raw.count(b'{')+raw.count(b'}')
    counts['variation_characters']+=raw.count(b'(')+raw.count(b')')
print(json.dumps({'raw_census':counts,'member_sha256':digest.hexdigest(),'scope':'raw Event-header census, separate from source-frame legality acceptance or deduplicated game counts'},indent=2,sort_keys=True))
