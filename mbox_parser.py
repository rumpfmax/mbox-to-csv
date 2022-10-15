from cgitb import text
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from email_reply_parser import EmailReplyParser
from email.utils import parsedate_tz, mktime_tz

import ast
import datetime
import mailbox
import ntpath
import os
import quopri
import re
# import rules
import sys
import time
import unicodecsv as csv
import spacy
import numpy as np

# converts seconds since epoch to mm/dd/yyyy string
def get_date(second_since_epoch, date_format):
    if second_since_epoch is None:
        return None
    time_tuple = parsedate_tz(email["date"])
    utc_seconds_since_epoch = mktime_tz(time_tuple)
    datetime_obj = datetime.datetime.fromtimestamp(utc_seconds_since_epoch)
    return datetime_obj.strftime(date_format)

# clean content
def clean_content(content):
    # decode message from "quoted printable" format
    content = quopri.decodestring(content)

    # try to strip HTML tags
    # if errors happen in BeautifulSoup (for unknown encodings), then bail
    try:
        text = BeautifulSoup(content, "lxml").text
    except Exception as e:
        return ''
    return text


# get contents of email
def get_content(email):
    parts = []

    for part in email.walk():
        if part.get_content_maintype() == 'multipart':
            continue

        content = part.get_payload(decode=True)
        part_contents = ""
        if content is None:
            part_contents = ""
        else:
            part_contents = EmailReplyParser.parse_reply(clean_content(content))

        parts.append(part_contents)

    return parts[0]

# get all emails in field
def get_emails_clean(field):
    # find all matches with format <user@example.com> or user@example.com
    matches = re.findall(r'\<?([a-zA-Z0-9_\-\.]+@[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,5})\>?', str(field))
    if matches:
        emails_cleaned = []
        for match in matches:
            emails_cleaned.append(match.lower())
        unique_emails = list(set(emails_cleaned))
        return sorted(unique_emails, key=str.lower)
    else:
        return []

# entry point
if __name__ == '__main__':

    # load environment settings
    load_dotenv(verbose=True)

    mbox_file = "example.mbox"
    file_name = ntpath.basename(mbox_file).lower()
    export_file_name = mbox_file + ".csv"
    export_file = open(export_file_name, "wb")

    # create CSV with header row
    writer = csv.writer(export_file, encoding='utf-8')
    writer.writerow(["index", "date", "from_addr", "to_addr", "cc_addr", "subject", "content"])

    # create row count
    row_written = 0
    emails = mailbox.mbox(mbox_file)
    print(len(emails))
    nlp = spacy.load('en_core_web_sm')


    for idx, email in enumerate(emails):
        # capture default content
        if email["date"] is None:
            continue
        date = mktime_tz(parsedate_tz(email["date"]))
        sent_from = get_emails_clean(email["from"])
        sent_to = get_emails_clean(email["to"])
        cc = get_emails_clean(email["cc"])
        subject = re.sub('[\n\t\r]', ' -- ', str(email["subject"])).lower()
        contents = get_content(email).lower()
        if contents == '':
            print(f"{idx}: {contents}")
            continue

        # we split passages meaningfully
        doc = nlp(contents)
        sents = list(doc.sents)
        vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

        threshold = 0.3 # controls sensitivity

        clusters = [[0]]
        for i in range(1, len(sents)):
            if np.dot(vecs[i], vecs[i-1]) < threshold:
                # here we use only the similarity between neighboring pairs of sentences.
                # instead, we can use the "weakest link" or "strongest link" approach.
                # potentially, it could improve the quality of clustering.
                clusters.append([])
            clusters[-1].append(i)

        pass_idx = 0
        for cluster in clusters:
            passage = ' '.join([sents[i].text for i in cluster])
            if len(passage) < 5:
                continue

            row = [f"{idx}_{pass_idx}", date, ", ".join(sent_from), ", ".join(sent_to), ", ".join(cc), subject, passage]

            # write the row
            writer.writerow(row)
            pass_idx += 1
            row_written += 1
            if row_written % 10000 == 0:
                print(f"Rows written: {row_written}")

    export_file.close()
