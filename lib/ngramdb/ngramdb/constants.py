import json

# myria server and port
REST_URL = "rest.myria.cs.washington.edu"
REST_PORT = 1776

# relation names in Myria
NGRAM_RELATION = "\"lzilles:ngrams:ngram\""
TOKEN_RELATION = "\"lzilles:ngrams:token\""
DEP_RELATION = "\"lzilles:ngrams:dependency\""
POS_RELATION = "\"lzilles:ngrams:partofspeech\""
TT_RELATION = "\"lzilles:ngrams:ngram_token_token\""

# query limitations
MIN_WORD_LEN = 3
MIN_WORD_COUNT = 1

# top-level sql query templates
SQL_CONTEXT_TEMPLATE = """SELECT
    t.nid, nginfo.freq, t.position, t.surface, pos.postag, dep.deprel,
    t.headposition
FROM
    "lzilles:ngrams:token" t,
    "lzilles:ngrams:dependency" dep,
    "lzilles:ngrams:partofspeech" pos,
    ({subquery}) AS nginfo

WHERE
    nginfo.nid=t.nid
    AND pos.posid=t.posid
    AND dep.depid=t.depid

ORDER BY
    nginfo.freq DESC,
    (t.nid, t.position) ASC;
"""

SQL_COUNT_TEMPLATE = """SELECT SUM(nginfo.freq) FROM ({subquery}) AS nginfo;"""

# json query plan templates
JSON_COUNT_TEMPLATE = json.loads("""
{
    "fragments": [
        {
            "operators": [
                {
                    "opId": 0,
                    "opType": "DbQueryScan",

                    "schema": {
                        "columnNames": [
                            "sum"
                        ],
                        "columnTypes": [
                            "LONG_TYPE"
                        ]
                    }

                },
                {
                    "opId": 1,
                    "argChild": 0,
                    "opType": "CollectProducer"
                }
            ]
        },
        {
            "operators": [
                {
                    "opId": 2,
                    "opType": "CollectConsumer",
                    "argOperatorId": 1
                },
                {
                    "opId": 3,
                    "opType": "Aggregate",
                    "argChild": 2,
                    "aggregators": [
                        {
                            "type": "SingleColumn",
                            "column": 0,
                            "aggOps": ["SUM"]
                        }
                    ]
                },
                {
                    "opId": 4,
                    "argChild": 3,
                    "opType": "DbInsert",
                    "argOverwriteTable": true,
                    "relationKey": {
                        "programName": "ngramoutput",
                        "relationName": "TEMPOUTCOUNT",
                        "userName": "lzilles"
                    }
                }
            ]
        }
    ],

    "logicalRa": "",
    "rawQuery": "[ ngram count test ]",
    "language": "sql"
}
""")

JSON_CONTEXT_TEMPLATE = json.loads("""
{
    "fragments": [
        {
            "operators": [
                {
                    "opId": 0,
                    "opType": "DbQueryScan",

                    "schema": {
                        "columnNames": [
                            "nid",
                            "freq",
                            "position",
                            "surface",
                            "postag",
                            "deprel",
                            "headposition"
                        ],
                        "columnTypes": [
                            "LONG_TYPE",
                            "INT_TYPE",
                            "INT_TYPE",
                            "STRING_TYPE",
                            "STRING_TYPE",
                            "STRING_TYPE",
                            "INT_TYPE"
                        ]
                    }

                },
                {
                    "opId": 1,
                    "argChild": 0,
                    "opType": "CollectProducer"
                }
            ]
        },
        {
            "operators": [
                {
                    "opId": 2,
                    "opType": "CollectConsumer",
                    "argOperatorId": 1
                },
                {
                    "opId": 3,
                    "opType": "InMemoryOrderBy",
                    "opName": "InMemSort(results)",
                    "argChild": 2,
                    "argSortColumns": [
                        1,
                        0,
                        2
                    ],
                    "argAscending": [
                        false,
                        true,
                        true
                    ]
                },
                {
                    "opId": 4,
                    "argChild": 3,
                    "opType": "DbInsert",
                    "argOverwriteTable": true,
                    "relationKey": {
                        "programName": "ngramoutput",
                        "relationName": "TEMPOUT",
                        "userName": "lzilles"
                    }
                }
            ]
        }
    ],

    "logicalRa": "",
    "rawQuery": "[ ngram test ]",
    "language": "sql"
}
""")
