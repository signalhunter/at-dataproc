import argparse
import orjson
import pendulum
import logging
import os
import sys
from clickhouse_driver import Client
from io import BytesIO
from fastwarc.warc import ArchiveIterator, WarcRecordType
from fastwarc.stream_io import GZipStream, FileStream
from ytdlp_utils import extract_comment, try_get, get_first, traverse_obj

class ClickhouseBulkInsert:
    def __init__(self, client, batchsize) -> None:
        self.buffer = []
        self.client = client
        self.batchsize = batchsize
        pass
    def insert(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.batchsize:
            self.write()
    def write(self):
        logging.info("Writing to database")
        client.execute("INSERT INTO youtube_discussions VALUES", self.buffer)
        self.buffer = []

parser = argparse.ArgumentParser(description="YouTube Discussions WARC Processor")
parser.add_argument("--dbhost", default="localhost", help="Clickhouse server address")
parser.add_argument("--dbport", type=int, default=9000, help="Clickhouse server port")
parser.add_argument("--dbuser", default="default", help="Clickhouse user")
parser.add_argument("--batch", type=int, default=10000, help="Clickhouse batch size")
parser.add_argument("warc", help="Path to WARC file")
parser.add_argument("log", help="Path to log file")
args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(args.log),
        logging.StreamHandler()
    ]
)

CLICKHOUSE_PASSWD = os.getenv("CLICKHOUSE_PASSWD")
if not CLICKHOUSE_PASSWD:
    logging.error("CLICKHOUSE_PASSWD env. variable not defined")
    sys.exit(1)

client = Client(
    host=args.dbhost,
    port=args.dbport,
    user=args.dbuser,
    password=CLICKHOUSE_PASSWD
)
clickhouse = ClickhouseBulkInsert(client, args.batch)

logging.info(f"Extracting {args.warc}")

req_count, com_count, hsc_count = 0, 0, 0
stream = GZipStream(FileStream(args.warc, "rb"))
for rec in ArchiveIterator(stream, record_types=WarcRecordType.response):
    if rec.http_headers.status_code != 200:
        logging.warning(f"Non-200 response {rec.http_headers.status_code} (req #{req_count+1})")
        hsc_count += 1
        continue
    req_count += 1
    if rec.http_headers.get("Transfer-Encoding") == "chunked":
        buffer = BytesIO()
        while True:
            clen = int(rec.reader.readline().strip(), 16)
            if clen != 0:
                buffer.write(rec.reader.read(clen))
            rec.reader.readline()
            if clen == 0:
                body = buffer.getbuffer()
                break
    else:
        body = rec.reader.read()
    
    data = orjson.loads(body)

    channel = rec.headers.get("X-Wget-AT-Project-Item-Name").split(":")[1]
    arctime = pendulum.parse(rec.headers.get("WARC-Date"))

    items = traverse_obj(
        data["onResponseReceivedEndpoints"][-1], # Hopefully there won't be a 3rd element
        (('reloadContinuationItemsCommand', 'appendContinuationItemsAction'), 'continuationItems'),
        get_all=False, expected_type=list) or []

    for i in items:
        thread_renderer = try_get(i, lambda x: x['commentThreadRenderer'])
        comment_renderer = get_first(
            (thread_renderer, i), [['commentRenderer', ('comment', 'commentRenderer')]],
            expected_type=dict, default={})
        comment = extract_comment(comment_renderer, arctime)
        if comment:
            com_count += 1
            clickhouse.insert((channel, *comment))

    if req_count % 2500 == 0:
        logging.info(f"Stats: {req_count} responses, {com_count} comments, {hsc_count} non-200 responses")

# Write any leftovers to database
clickhouse.write()
logging.info(f"Finished - {req_count} responses, {com_count} comments, {hsc_count} non-200 responses")
