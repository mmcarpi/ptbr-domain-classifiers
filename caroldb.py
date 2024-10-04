#!/usr/bin/env python
# coding: utf-8
import json
import tarfile
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import polars as pl
import requests
from lxml import etree


def teitag(suffix):
    return "{http://www.tei-c.org/ns/1.0}" + suffix


def extract_texts_from_xml(path, namespace, parseargs):
    output = []
    for _, teidoc in etree.iterparse(path, **parseargs):
        value = dict()
        for ref in teidoc.findall(f".//{namespace}catRef[@target]"):
            scheme, target = ref.get("scheme"), ref.get("target")
            value[ref.get("scheme")[1:].lower()] = ref.get("target")
        value["domain"] = "".join(
            e.text for e in teidoc.findall(f".//{namespace}domain")
        )
        value["text"] = " ".join(
            e.text for e in teidoc.findall(f".//{namespace}body/{namespace}p") if e.text
        )
        output.append(value)
    return output


def load_caroldb_from_disk(caroldb_path):
    parseargs = dict(huge_tree=True, encoding="utf-8", tag=teitag("TEI"))
    namespace = teitag("")
    xmls = [f for p in caroldb_path.iterdir() for f in p.iterdir()]
    table = defaultdict(list)
    with Pool(processes=4) as pool:
        for result in pool.map(
            partial(extract_texts_from_xml, namespace=namespace, parseargs=parseargs),
            xmls,
        ):
            for value in result:
                for k, v in value.items():
                    table[k].append(v)

    return pl.from_dict(table)


def get_caroldb(caroldb_url, caroldb_tar_path):
    response = requests.get(caroldb_url, params={"download": "true"})
    if not response.ok:
        print(
            "Error trying to download. Please download manually from %s" % caroldb_url
        )
        print("After that, run the script again to finish dataset creation.")
        exit(1)

    with open(caroldb_tar_path, "wb") as targzfile:
        targzfile.write(response.content)


def main():
    caroldb_path = Path("Data/CarolDB/")
    caroldb_tar_path = Path("Data/CarolDB.tar.gz")
    caroldb_parquet_path = Path("Data/caroldb.parquet")
    caroldb_url = "https://media.githubusercontent.com/media/marianasturzeneker/SubcorporaCarolina/main/Corpora/CarolDB.tar.gz"

    if not caroldb_parquet_path.exists():
        if not caroldb_tar_path.exists():
            print(
                f"Downloading CarolDB into {caroldb_tar_path}. This may take a while."
            )
            get_caroldb(caroldb_url, caroldb_tar_path)

        if not caroldb_path.exists():
            print(f"Extracting {caroldb_tar_path}")
            with tarfile.open(caroldb_tar_path) as tf:
                tf.extractall(path=caroldb_path, filter="data")

        print(f"Writing {caroldb_parquet_path}")
        df = load_caroldb_from_disk(caroldb_path / "Carolina_dedupe")
        df.write_parquet(caroldb_parquet_path, compression=None)
    else:
        print(f"{caroldb_parquet_path} exists.")


if __name__ == "__main__":
    main()
