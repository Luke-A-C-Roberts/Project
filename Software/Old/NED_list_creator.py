# imports

# requests module allows use of http GET requests to scrape from the NED database
from requests import get
from requests.models import Response

# bs4 is a web scraping parser for disecting html
from bs4 import BeautifulSoup

# pandas for manipulating dataframes
from pandas import DataFrame, concat as df_concat

# other standard library utilities
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import starmap
from re import search as regex_search
from typing import Mapping
import logging


NED_URL_ROOT: str = "https://ned.ipac.caltech.edu/cgi-bin/OBJatt?"

DEFAULT_SEARCH_VALS: list[tuple[str, str]] = [
    ("delimeter", "bar"),
    ("NO_LINKS", "1"),
    ("crosid", "objname"),
    ("position", "z"),
    ("gadata", "morphol"),
]

DF_COLS: list[str] = ["Name", "Redshift", "Classification"]

MORPHOLOGY_CODES: dict[str, tuple[str, str]] = {
    "E": ("M", "1010"),
    "S0": ("M", "1234"),
    "S0/a": ("M", "1250"),
    "S": ("M", "1168"),  # This class encompasses A B and AB so might not use
    "Irr": ("M", "1157"),  # Not sure about irregulars as they are a minority class
    "A": ("M", "1005"),
    "AB": ("M", "1006"),
    "B": ("M", "1008"),
    # advanced
    "SAa": [("M", "1168"), ("M", "1327"), ("M", "2302")],
    "SAab": [("M", "1168"), ("M", "1327"), ("M", "2312")],
    "SAb": [("M", "1168"), ("M", "1327"), ("M", "2322")],
    "SAbc": [("M", "1168"), ("M", "1327"), ("M", "2332")],
    "SAc": [("M", "1168"), ("M", "1327"), ("M", "2341")],
    "SAcd": [("M", "1168"), ("M", "1327"), ("M", "2351")],
    "SAd": [("M", "1168"), ("M", "1327"), ("M", "2360")],
    "SABa": [("M", "1168"), ("M", "1576"), ("M", "2302")],
    "SABab": [("M", "1168"), ("M", "1576"), ("M", "2312")],
    "SABb": [("M", "1168"), ("M", "1576"), ("M", "2322")],
    "SABbc": [("M", "1168"), ("M", "1576"), ("M", "2332")],
    "SABc": [("M", "1168"), ("M", "1576"), ("M", "2341")],
    "SABcd": [("M", "1168"), ("M", "1576"), ("M", "2351")],
    "SABd": [("M", "1168"), ("M", "1576"), ("M", "2360")],
    "SBa": [("M", "1168"), ("M", "1924"), ("M", "2302")],
    "SBab": [("M", "1168"), ("M", "1924"), ("M", "2312")],
    "SBb": [("M", "1168"), ("M", "1924"), ("M", "2322")],
    "SBbc": [("M", "1168"), ("M", "1924"), ("M", "2332")],
    "SBc": [("M", "1168"), ("M", "1924"), ("M", "2341")],
    "SBcd": [("M", "1168"), ("M", "1924"), ("M", "2351")],
    "SBd": [("M", "1168"), ("M", "1924"), ("M", "2360")],
}


def make_ned_search_url(values: list[tuple[str, str]]) -> str:
    return NED_URL_ROOT + "&".join(starmap(lambda arg, val: f"{arg}={val}", values))


def request_page(url: str) -> Response:
    response = get(url)
    return response


def find_page_count(string: str) -> int:
    soup = BeautifulSoup(string, "lxml")
    form = soup.find("form")
    text = form.get_text().replace("\xa0", "").strip()
    page_numbers = regex_search(r"Page [0-9]+ of [0-9]+", text).group(0)
    return int(regex_search(r"[0-9]+$", page_numbers).group(0))


# Help:
#   https://stackoverflow.com/questions/50633050/scrape-tables-into-dataframe-with-beautifulsoup
#   https://www.geeksforgeeks.org/how-to-remove-tags-using-beautifulsoup-in-python/
def parse_page(response: Response) -> list[str]:
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table")
    table_rows = table.find("pre")
    table_rows.find("strong").decompose()
    return table_rows.get_text().splitlines()[1:]


def extract_data(table_rows: list[str]) -> list[Mapping]:
    return [map(lambda item: item.strip(), string.split("|")) for string in table_rows]


def data_to_df(table_rows: list[Mapping]) -> DataFrame:
    df = DataFrame(columns=DF_COLS)
    for row in table_rows:
        df_row = {col: val for (col, val) in zip(DF_COLS, row)}
        df.loc[len(df)] = df_row
    return df


def pipeline(*funcs: list) -> any:
    val = funcs[0]
    for func in funcs[1:]:
        val = func(val)
    return val


def request_pipeline(url: str) -> DataFrame:
    return pipeline(url, request_page, parse_page, extract_data, data_to_df)


def request_all_pages(url: str, pages: int) -> DataFrame:
    if pages == 1:
        return request_pipeline(url)

    urls = map(lambda page: f"{url}&page={page}", range(1, pages + 1))
    responses = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_url_requests = {
            executor.submit(request_pipeline, url): url for url in urls
        }

        for future in as_completed(future_url_requests):
            url = future_url_requests[future]
            try:
                responses.append(future.result())
                logging.info(f"Successfully requested and parsed {url}")
            except BaseException as e:
                logging.error(
                    f"Bad response from response or parsing of {url}:\n\n{str(e)}"
                )

    return df_concat(responses).sort_values("Name", ignore_index=True)


def page_count_pipeline(url: str) -> int:
    return pipeline(url, request_page, lambda res: res.text, find_page_count)


def morphology_search(classification: str) -> DataFrame:
    search_vals = DEFAULT_SEARCH_VALS + [MORPHOLOGY_CODES[classification]]
    url = make_ned_search_url(search_vals)
    page_count = page_count_pipeline(url)
    all_pages = request_all_pages(url, page_count)
    all_pages["Requested"] = classification
    return all_pages


def main() -> None:
    logger = logging.Logger("Logger")
    logger.setLevel(logging.INFO)

    morphologies = ["E", "S0", "S0/a", "A", "AB", "B"]
    queried_morphologies = map(morphology_search, morphologies)
    galaxy_df = (
        df_concat([*queried_morphologies])
        .sort_values("Name", ignore_index=True)
        .dropna()
    )

    galaxy_df.to_csv("./NED_list.csv", sep=",", index=False)


if __name__ == "__main__":
    main()
