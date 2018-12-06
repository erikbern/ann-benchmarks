This document specifies a simple text-based protocol that can be used to benchmark algorithms that don't have a Python wrapper. A program that implements the algorithm side of this specification will be referred to in the rest of this document as a "front-end".

This protocol is line-oriented; both sides should configure their input and output streams to be line-buffered. Front-ends receive messages by reading lines from standard input and send messages by writing lines to standard output.

## Modes

A front-end begins in configuration mode. When configuration is complete, it transitions into training mode; when training data has been supplied, into query mode; and, when no more queries remain, it terminates. It isn't possible to return from one mode to an earlier mode without restarting the front-end.

A front-end reads lines from standard input, tokenises them, and interprets them according to its current mode; responses are written as lines to standard output. To enable protocol responses to be distinguished from other messages that may appear on standard output, the first token of a line containing a response will always be `epbprtv0`; the second will be `ok` when a command succeeds, potentially followed by other tokens, and `fail` when it doesn't.

(The obscure token `epbprtv0` is intended to uniquely identify this protocol, and is meant to suggest something like "**e**xternal **p**rogram **b**enchmarking **pr**o**t**ocol, **v**ersion **0**".)

A front-end may choose to include extra tokens in its responses after the tokens required by this specification to communicate more information back to the caller.

## Tokenisation

Both the front-end and `ann-benchmarks` perform *tokenisation* on the lines of text they send and receive. The rules for tokenisation are as follows:

* A token is a sequence of characters separated by one or more whitespace characters.

  Input | Token 1 | Token 2 | Token 3
  ----- | ------- | ------- | -------
  abc | abc | |
  a bc | a | bc |
  a    bc | a | bc |
  a b c | a | b | c

* A sequence surrounded by single quote marks will be treated as part of a token, even if it contains whitespace or doesn't contain any other characters.

  Input | Token 1 | Token 2 | Token 3
  ----- | ------- | ------- | -------
  'a b c' | a b c | |
  'a b c'd | a b cd | |
  a '' b | a | *empty string* | b

* A sequence surrounded by double quote marks will be treated as part of a token, even if it contains whitespace or doesn't contain any other characters.

  Input | Token 1 | Token 2 | Token 3
  ----- | ------- | ------- | -------
  "a b c" | a b c | |
  "a b c"d | a b cd | |
  a "" b | a | *empty string* | b

* Outside of a quoted sequence, preceding a character with a backslash causes any special significance it may have to be ignored; the character is then said to have been "escaped".

  Input | Token 1 | Token 2 | Token 3
  ----- | ------- | ------- | -------
  \a \b \c | a | b | c

  An escaped whitespace character doesn't separate tokens:

  Input | Token 1 | Token 2
  ----- | ------- | -------
  a b\ c | a | b c |
  "a b c"\ d | a b c d |

  An escaped quote mark doesn't begin a sequence:

  Input | Token 1 | Token 2 | Token 3
  ----- | ------- | ------- | -------
  \'a b c\' | a | b | c |
  \"a b c\" | a | b | c |

  An escaped backslash doesn't escape the subsequent character:

  Input | Token 1 | Token 2
  ----- | ------- | -------
  a\\\\"b c" d | a\b c | d

* In sequences begun by a double quote mark, only double quote marks and backslashes (and, for compatibility reasons, dollar signs) may be escaped; the backslash otherwise has no special significance.

  Input | Token 1 | Token 2 | Token 3
  ----- | ------- | ------- | -------
  "\a \b" \c | \a \b | c |
  "\\\\ \\" \\$ a" "\b" c | \ " $ a | \b | c

* In sequences begun by a single quote mark, a backslash has no special significance.

  Input | Token 1 | Token 2
  ----- | ------- | -------
  'a b' c | a b | c
  'a b\\' c | a b\ | c

Apart from the fact that newline characters can't be escaped, these rules should match the tokenisation rules of the POSIX shell.

## Commands

Commands are sent to the front-end by `ann-benchmarks`. Each command consists of a single line of text; the front-end replies with one or more lines of text. Front-ends can't initiate communication; they can only reply to commands.

This section specifies these commands, along with the possible responses a front-end might send.

If a front-end receives a command that it doesn't understand in the current mode (or at all), it should respond with `epbprtv0 fail` and continue processing commands.

### Configuration mode

In configuration mode, front-ends should respond to three different kinds of command:

#### `VAR VAL` (two tokens)

Set the value of the algorithm configuration option `VAR` to `VAL`.

Responses:

* `epbprtv0 ok`

  The value specified for the algorithm configuration option `VAR` was acceptable, and the option has been set.

* `epbprtv0 fail`

  The value specified for the algorithm configuration option `VAR` wasn't acceptable. No change has been made to the value of this option.

#### `frontend VAR VAL` (three tokens)

Set the value of the front-end configuration option `VAR` to `VAL`. Front-end configuration options may cause the front-end to behave in a manner other than that described in this specification.

Responses:

* `epbprtv0 ok`

  The value specified for the front-end configuration option `VAR` was acceptable, and the option has been set.

* `epbprtv0 fail`

  The value specified for the front-end configuration option `VAR` wasn't acceptable. No change has been made to the value of this option.

#### *empty line* (zero tokens)

Finish configuration mode and enter training mode.

Responses:

* `epbprtv0 ok`

  Training mode has been entered.

* `epbprtv0 fail`

  One or more configuration options required by the algorithm weren't specified, and so the query process has terminated.

### Training mode

In training mode, front-ends should respond to two different kinds of command:

#### `ENTRY` (one token)

Interpret `ENTRY` as an item of training data.

Responses:

* `epbprtv0 ok`

  `ENTRY` was added as the next item of training data. The index values returned in query mode refer to the first item added as `0`, the second as `1`, and so on.

* `epbprtv0 fail`

  Either `ENTRY` couldn't be interpreted as an item of training data, or the training data wasn't accepted.

#### *empty line* (zero tokens)

Finish training mode and enter query mode.

Responses:

* `epbprtv0 ok COUNT1 [fail COUNT2]`

  `COUNT1` (potentially zero) entries were successfully interpreted and added to the data structure. (`COUNT2` entries couldn't be interpreted or couldn't be added for other reasons.)

### Query mode

In query mode, front-ends should respond to two different kinds of command:

#### `ENTRY N` (two tokens)

Return the indices of at most `N` (greater than or equal to 1) close matches for `ENTRY`.

Responses:

* `epbprtv0 ok R`

  `R` (greater than zero and less than or equal to `N`) close matches were found. Each of the next `R` lines, when tokenised, will consist of the token `epbprtv0` followed by a token specifying the index of a close match. (The first line should identify the *closest* close match, and the `R`-th should identify the furthest away.)

* `epbprtv0 fail`

  No close matches were found.

#### *empty line* (zero tokens)

Finish query mode and terminate the front-end.

Responses:

* `epbprtv0 ok`

  The front-end has terminated.
