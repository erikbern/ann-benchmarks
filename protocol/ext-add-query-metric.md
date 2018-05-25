(This document describes an extension that front-ends aren't required to implement. In fact, no front-end is *known* to implement it; this document serves as an example of how to extend the protocol. Front-ends that don't implement this extension should reject attempts to set the `add-query-metric` configuration option.)

When the configuration option `add-query-metric` is set to a value other than `all`, if that value identifies a query metric known to the front-end, then the value for this metric will be appended to each query response. This option may be set several times; each one will (try to) add another query metric.

Setting this option to the value `all` will cause *all* metrics known to the front-end to be included.

## Commands

### Configuration mode

#### `add-query-metric METRIC` (two tokens)

Request that query responses also include the value of the query metric `METRIC`, if that's recognised by the front-end.

Responses:

* `epbprtv0 ok`

  The metric `METRIC` was recognised, and query responses will include a value for it.

* `epbprtv0 fail`

  The metric `METRIC` was not recognised; query responses will not include a value for it.

#### `add-query-metric all` (two tokens)

Request that query responses also include the values of all query metrics recognised by the front-end.

Responses:

* `epbprtv0 ok`

  Query responses will include the values of all metrics known to the front-end. (This may not actually change the output; the front-end could, in principle, support this extension but not recognise any query metrics.)

* `epbprtv0 fail`

  Front-ends may choose to emit this response if they do not recognise *any* query metrics, but they may also emit `epbprtv0 ok` in these circumstances (to indicate that all zero metrics will be included in the output).

### Query mode

#### `ENTRY N` (two tokens)

This extension changes the behaviour of one response:

* `epbprtv0 ok R [NAME0 VALUE0 ...]`

  `R` (greater than zero and less than or equal to `N`) close matches were found. Each of the next `R` lines, when tokenised, will consist of the token `epbprtv0` followed by a token specifying the index of a close match. (The first line should identify the *closest* close match, and the `R`-th should identify the furthest away.)

  If additional query metrics were specified and recognised during configuration mode, then their names and values will be provided as a number of pairs of tokens after `R`. For example, a response including the hypothetical `buckets_searched` and `candidates_checked` metrics might look like this:
  
  `epbprtv0 ok 10 buckets_searched 8 candidates_checked 507`
