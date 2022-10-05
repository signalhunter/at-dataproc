# Taken from yt-dlp's utils.py, common.py, and youtube.py

import calendar
import collections
import collections.abc
import contextlib
import functools
import itertools
import pendulum
import re
from pendulum.datetime import DateTime

NO_DEFAULT = object()
IDENTITY = lambda x: x

TIMEZONE_NAMES = {
    'UT': 0, 'UTC': 0, 'GMT': 0, 'Z': 0,
    'AST': -4, 'ADT': -3,  # Atlantic (used in Canada)
    'EST': -5, 'EDT': -4,  # Eastern
    'CST': -6, 'CDT': -5,  # Central
    'MST': -7, 'MDT': -6,  # Mountain
    'PST': -8, 'PDT': -7   # Pacific
}

DATE_FORMATS = (
    '%d %B %Y',
    '%d %b %Y',
    '%B %d %Y',
    '%B %dst %Y',
    '%B %dnd %Y',
    '%B %drd %Y',
    '%B %dth %Y',
    '%b %d %Y',
    '%b %dst %Y',
    '%b %dnd %Y',
    '%b %drd %Y',
    '%b %dth %Y',
    '%b %dst %Y %I:%M',
    '%b %dnd %Y %I:%M',
    '%b %drd %Y %I:%M',
    '%b %dth %Y %I:%M',
    '%Y %m %d',
    '%Y-%m-%d',
    '%Y.%m.%d.',
    '%Y/%m/%d',
    '%Y/%m/%d %H:%M',
    '%Y/%m/%d %H:%M:%S',
    '%Y%m%d%H%M',
    '%Y%m%d%H%M%S',
    '%Y%m%d',
    '%Y-%m-%d %H:%M',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d %H:%M:%S:%f',
    '%d.%m.%Y %H:%M',
    '%d.%m.%Y %H.%M',
    '%Y-%m-%dT%H:%M:%SZ',
    '%Y-%m-%dT%H:%M:%S.%fZ',
    '%Y-%m-%dT%H:%M:%S.%f0Z',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%dT%H:%M',
    '%b %d %Y at %H:%M',
    '%b %d %Y at %H:%M:%S',
    '%B %d %Y at %H:%M',
    '%B %d %Y at %H:%M:%S',
    '%H:%M %d-%b-%Y',
)

DATE_FORMATS_DAY_FIRST = list(DATE_FORMATS)
DATE_FORMATS_DAY_FIRST.extend([
    '%d-%m-%Y',
    '%d.%m.%Y',
    '%d.%m.%y',
    '%d/%m/%Y',
    '%d/%m/%y',
    '%d/%m/%Y %H:%M:%S',
    '%d-%m-%Y %H:%M',
])

DATE_FORMATS_MONTH_FIRST = list(DATE_FORMATS)
DATE_FORMATS_MONTH_FIRST.extend([
    '%m-%d-%Y',
    '%m.%d.%Y',
    '%m/%d/%Y',
    '%m/%d/%y',
    '%m/%d/%Y %H:%M:%S',
])

# youtube.py
def extract_comment(comment_renderer, retrieval_time):
    comment_id = comment_renderer.get('commentId')
    if not comment_id:
        return

    text = get_text(comment_renderer, 'contentText')

    # Timestamp is an estimate calculated from the current time and time_text
    time_text = get_text(comment_renderer, 'publishedTimeText') or ''
    timestamp = parse_time_text(time_text, retrieval_time)

    author = get_text(comment_renderer, 'authorText')
    author_id = try_get(comment_renderer,
                        lambda x: x['authorEndpoint']['browseEndpoint']['browseId'], str)

    votes = parse_count(try_get(comment_renderer, (lambda x: x['voteCount']['simpleText'],
                                                    lambda x: x['likeCount']), str)) or 0
    author_thumbnail = try_get(comment_renderer,
                                lambda x: x['authorThumbnail']['thumbnails'][-1]['url'], str)

    is_favorited = 'creatorHeart' in (try_get(
        comment_renderer, lambda x: x['actionButtons']['commentActionButtonsRenderer'], dict) or {})

    # id, author_id, author, timestamp, like_count, is_favorited, text, author_thumb
    return (
        comment_id,
        author_id,
        author,
        int(timestamp),
        votes,
        is_favorited,
        text,
        author_thumbnail.split("=")[0]
    )

def get_text(data, *path_list, max_runs=None):
    for path in path_list or [None]:
        if path is None:
            obj = [data]
        else:
            obj = traverse_obj(data, path, default=[])
            if not any(key is ... or isinstance(key, (list, tuple)) for key in variadic(path)):
                obj = [obj]
        for item in obj:
            text = try_get(item, lambda x: x['simpleText'], str)
            if text:
                return text
            runs = try_get(item, lambda x: x['runs'], list) or []
            if not runs and isinstance(item, list):
                runs = item

            runs = runs[:min(len(runs), max_runs or len(runs))]
            text = ''.join(traverse_obj(runs, (..., 'text'), expected_type=str, default=[]))
            if text:
                return text

def parse_time_text(text, time):
    if not text:
        return
    dt = extract_relative_time(text, time)
    timestamp = None
    if isinstance(dt, DateTime):
        timestamp = dt.timestamp()

    # probably don't need to handle this edge case for comments
    # if timestamp is None:
    #     timestamp = (
    #         unified_timestamp(text) or unified_timestamp(
    #             search_regex(
    #                 (r'([a-z]+\s*\d{1,2},?\s*20\d{2})', r'(?:.+|^)(?:live|premieres|ed|ing)(?:\s*(?:on|for))?\s*(.+\d)'),
    #                 text.lower(), 'time text', default=None)))

    if text and timestamp is None:
        raise RuntimeError(f'Cannot parse localized time text "{text}"')
    return timestamp

def extract_relative_time(relative_time_text, time):
    mobj = re.search(r'(?P<start>today|yesterday|now)|(?P<time>\d+)\s*(?P<unit>microsecond|second|minute|hour|day|week|month|year)s?\s*ago', relative_time_text)
    if mobj:
        start = mobj.group('start')
        if start:
            return datetime_from_str(start, ts=time)
        try:
            return datetime_from_str('now-%s%s' % (mobj.group('time'), mobj.group('unit')), ts=time)
        except ValueError:
            return None

# common.py
#def search_regex(pattern, string, name, default=NO_DEFAULT, fatal=True, flags=0, group=None):
#    if string is None:
#        mobj = None
#    elif isinstance(pattern, (str, re.Pattern)):
#        mobj = re.search(pattern, string, flags)
#    else:
#        for p in pattern:
#            mobj = re.search(p, string, flags)
#            if mobj:
#                break
#
#    if mobj:
#        if group is None:
#            # return the first matching group
#            return next(g for g in mobj.groups() if g is not None)
#        elif isinstance(group, (list, tuple)):
#            return tuple(mobj.group(g) for g in group)
#        else:
#            return mobj.group(group)
#    elif default is not NO_DEFAULT:
#        return default
#    else:
#        raise RuntimeError("Unable to extract regex " + name)

# utils.py
def datetime_from_str(date_str, precision='auto', format='%Y%m%d', ts=None):
    auto_precision = False
    if precision == 'auto':
        auto_precision = True
        precision = 'microsecond'
    today = datetime_round(ts, precision)
    if date_str in ('now', 'today'):
        return today
    if date_str == 'yesterday':
        return today - pendulum.duration(days=1)
    match = re.match(
        r'(?P<start>.+)(?P<sign>[+-])(?P<time>\d+)(?P<unit>microsecond|second|minute|hour|day|week|month|year)s?',
        date_str)
    if match is not None:
        start_time = datetime_from_str(match.group('start'), precision, format, ts=ts)
        time = int(match.group('time')) * (-1 if match.group('sign') == '-' else 1)
        unit = match.group('unit')
        if unit == 'month' or unit == 'year':
            new_date = datetime_add_months(start_time, time * 12 if unit == 'year' else time)
            unit = 'day'
        else:
            if unit == 'week':
                unit = 'day'
                time *= 7
            delta = pendulum.duration(**{unit + 's': time})
            new_date = start_time + delta
        if auto_precision:
            return datetime_round(new_date, unit)
        return new_date

    return datetime_round(pendulum.from_format(date_str, format), precision)

# def unified_timestamp(date_str, day_first=True):
#     if date_str is None:
#         return None
#
#     date_str = re.sub(r'\s+', ' ', re.sub(
#         r'(?i)[,|]|(mon|tues?|wed(nes)?|thu(rs)?|fri|sat(ur)?)(day)?', '', date_str))
#
#     pm_delta = 12 if re.search(r'(?i)PM', date_str) else 0
#     timezone, date_str = extract_timezone(date_str)
#
#     # Remove AM/PM + timezone
#     date_str = re.sub(r'(?i)\s*(?:AM|PM)(?:\s+[A-Z]+)?', '', date_str)
#
#     # Remove unrecognized timezones from ISO 8601 alike timestamps
#     m = re.search(r'\d{1,2}:\d{1,2}(?:\.\d+)?(?P<tz>\s*[A-Z]+)$', date_str)
#     if m:
#         date_str = date_str[:-len(m.group('tz'))]
#
#     # Python only supports microseconds, so remove nanoseconds
#     m = re.search(r'^([0-9]{4,}-[0-9]{1,2}-[0-9]{1,2}T[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}\.[0-9]{6})[0-9]+$', date_str)
#     if m:
#         date_str = m.group(1)
#
#     for expression in date_formats(day_first):
#         with contextlib.suppress(ValueError):
#             dt = datetime.datetime.strptime(date_str, expression) - timezone + datetime.timedelta(hours=pm_delta)
#             return calendar.timegm(dt.timetuple())
#
#     timetuple = email.utils.parsedate_tz(date_str)
#     if timetuple:
#         return calendar.timegm(timetuple) + pm_delta * 3600 - timezone.total_seconds()

#def date_formats(day_first=True):
#    return DATE_FORMATS_DAY_FIRST if day_first else DATE_FORMATS_MONTH_FIRST

def datetime_round(dt, precision='day'):
    if precision == 'microsecond':
        return dt

    unit_seconds = {
        'day': 86400,
        'hour': 3600,
        'minute': 60,
        'second': 1,
    }
    roundto = lambda x, n: ((x + n / 2) // n) * n
    timestamp = dt.timestamp()
    return pendulum.from_timestamp(roundto(timestamp, unit_seconds[precision]))

def datetime_add_months(dt, months):
    month = dt.month + months - 1
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year, month, day)

#def extract_timezone(date_str):
#    m = re.search(
#        r'''(?x)
#            ^.{8,}?                                              # >=8 char non-TZ prefix, if present
#            (?P<tz>Z|                                            # just the UTC Z, or
#                (?:(?<=.\b\d{4}|\b\d{2}:\d\d)|                   # preceded by 4 digits or hh:mm or
#                   (?<!.\b[a-zA-Z]{3}|[a-zA-Z]{4}|..\b\d\d))     # not preceded by 3 alpha word or >= 4 alpha or 2 digits
#                   [ ]?                                          # optional space
#                (?P<sign>\+|-)                                   # +/-
#                (?P<hours>[0-9]{2}):?(?P<minutes>[0-9]{2})       # hh[:]mm
#            $)
#        ''', date_str)
#    if not m:
#        m = re.search(r'\d{1,2}:\d{1,2}(?:\.\d+)?(?P<tz>\s*[A-Z]+)$', date_str)
#        timezone = TIMEZONE_NAMES.get(m and m.group('tz').strip())
#        if timezone is not None:
#            date_str = date_str[:-len(m.group('tz'))]
#        timezone = pendulum.duration(hours=timezone or 0)
#    else:
#        date_str = date_str[:-len(m.group('tz'))]
#        if not m.group('sign'):
#            timezone = pendulum.duration()
#        else:
#            sign = 1 if m.group('sign') == '+' else -1
#            timezone = pendulum.duration(
#                hours=sign * int(m.group('hours')),
#                minutes=sign * int(m.group('minutes')))
#    return timezone, date_str

def parse_count(s):
    if s is None:
        return None

    s = re.sub(r'^[^\d]+\s', '', s).strip()

    if re.match(r'^[\d,.]+$', s):
        return str_to_int(s)

    _UNIT_TABLE = {
        'k': 1000,
        'K': 1000,
        'm': 1000 ** 2,
        'M': 1000 ** 2,
        'kk': 1000 ** 2,
        'KK': 1000 ** 2,
        'b': 1000 ** 3,
        'B': 1000 ** 3,
    }

    ret = lookup_unit_table(_UNIT_TABLE, s)
    if ret is not None:
        return ret

    mobj = re.match(r'([\d,.]+)(?:$|\s)', s)
    if mobj:
        return str_to_int(mobj.group(1))

def lookup_unit_table(unit_table, s):
    units_re = '|'.join(re.escape(u) for u in unit_table)
    m = re.match(
        r'(?P<num>[0-9]+(?:[,.][0-9]*)?)\s*(?P<unit>%s)\b' % units_re, s)
    if not m:
        return None
    num_str = m.group('num').replace(',', '.')
    mult = unit_table[m.group('unit')]
    return int(float(num_str) * mult)

def str_to_int(int_str):
    if isinstance(int_str, int):
        return int_str
    elif isinstance(int_str, str):
        int_str = re.sub(r'[,\.\+]', '', int_str)
        return int_or_none(int_str)

def variadic(x, allowed_types=(str, bytes, dict)):
    return x if isinstance(x, collections.abc.Iterable) and not isinstance(x, allowed_types) else (x,)

def try_get(src, getter, expected_type=None):
    return try_call(*variadic(getter), args=(src,), expected_type=expected_type)

def get_first(obj, keys, **kwargs):
    return traverse_obj(obj, (..., *variadic(keys)), **kwargs, get_all=False)

def try_call(*funcs, expected_type=None, args=[], kwargs={}):
    for f in funcs:
        try:
            val = f(*args, **kwargs)
        except (AttributeError, KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
            pass
        else:
            if expected_type is None or isinstance(val, expected_type):
                return val

def int_or_none(v, scale=1, default=None, get_attr=None, invscale=1):
    if get_attr and v is not None:
        v = getattr(v, get_attr, None)
    try:
        return int(v) * invscale // scale
    except (ValueError, TypeError, OverflowError):
        return default

def traverse_obj(
        obj, *paths, default=None, expected_type=None, get_all=True,
        casesense=True, is_user_input=False, traverse_string=False):

    is_sequence = lambda x: isinstance(x, collections.abc.Sequence) and not isinstance(x, (str, bytes))
    casefold = lambda k: k.casefold() if isinstance(k, str) else k

    if isinstance(expected_type, type):
        type_test = lambda val: val if isinstance(val, expected_type) else None
    else:
        type_test = lambda val: try_call(expected_type or IDENTITY, args=(val,))

    def apply_key(key, obj):
        if obj is None:
            return

        elif key is None:
            yield obj

        elif isinstance(key, (list, tuple)):
            for branch in key:
                _, result = apply_path(obj, branch)
                yield from result

        elif key is ...:
            if isinstance(obj, collections.abc.Mapping):
                yield from obj.values()
            elif is_sequence(obj):
                yield from obj
            elif traverse_string:
                yield from str(obj)

        elif callable(key):
            if is_sequence(obj):
                iter_obj = enumerate(obj)
            elif isinstance(obj, collections.abc.Mapping):
                iter_obj = obj.items()
            elif traverse_string:
                iter_obj = enumerate(str(obj))
            else:
                return
            yield from (v for k, v in iter_obj if try_call(key, args=(k, v)))

        elif isinstance(key, dict):
            iter_obj = ((k, _traverse_obj(obj, v)) for k, v in key.items())
            yield {k: v if v is not None else default for k, v in iter_obj
                   if v is not None or default is not None}

        elif isinstance(obj, dict):
            yield (obj.get(key) if casesense or (key in obj)
                   else next((v for k, v in obj.items() if casefold(k) == key), None))

        else:
            if is_user_input:
                key = (int_or_none(key) if ':' not in key
                       else slice(*map(int_or_none, key.split(':'))))

            if not isinstance(key, (int, slice)):
                return

            if not is_sequence(obj):
                if not traverse_string:
                    return
                obj = str(obj)

            with contextlib.suppress(IndexError):
                yield obj[key]

    def apply_path(start_obj, path):
        objs = (start_obj,)
        has_branched = False

        for key in variadic(path):
            if is_user_input and key == ':':
                key = ...

            if not casesense and isinstance(key, str):
                key = key.casefold()

            if key is ... or isinstance(key, (list, tuple)) or callable(key):
                has_branched = True

            key_func = functools.partial(apply_key, key)
            objs = itertools.chain.from_iterable(map(key_func, objs))

        return has_branched, objs

    def _traverse_obj(obj, path):
        has_branched, results = apply_path(obj, path)
        results = LazyList(x for x in map(type_test, results) if x is not None)
        if results:
            return results.exhaust() if get_all and has_branched else results[0]

    for path in paths:
        result = _traverse_obj(obj, path)
        if result is not None:
            return result

    return default

class LazyList(collections.abc.Sequence):
    class IndexError(IndexError):
        pass

    def __init__(self, iterable, *, reverse=False, _cache=None):
        self._iterable = iter(iterable)
        self._cache = [] if _cache is None else _cache
        self._reversed = reverse

    def __iter__(self):
        if self._reversed:
            # We need to consume the entire iterable to iterate in reverse
            yield from self.exhaust()
            return
        yield from self._cache
        for item in self._iterable:
            self._cache.append(item)
            yield item

    def _exhaust(self):
        self._cache.extend(self._iterable)
        self._iterable = []  # Discard the emptied iterable to make it pickle-able
        return self._cache

    def exhaust(self):
        """Evaluate the entire iterable"""
        return self._exhaust()[::-1 if self._reversed else 1]

    @staticmethod
    def _reverse_index(x):
        return None if x is None else ~x

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if self._reversed:
                idx = slice(self._reverse_index(idx.start), self._reverse_index(idx.stop), -(idx.step or 1))
            start, stop, step = idx.start, idx.stop, idx.step or 1
        elif isinstance(idx, int):
            if self._reversed:
                idx = self._reverse_index(idx)
            start, stop, step = idx, idx, 0
        else:
            raise TypeError('indices must be integers or slices')
        if ((start or 0) < 0 or (stop or 0) < 0
                or (start is None and step < 0)
                or (stop is None and step > 0)):
            # We need to consume the entire iterable to be able to slice from the end
            # Obviously, never use this with infinite iterables
            self._exhaust()
            try:
                return self._cache[idx]
            except IndexError as e:
                raise self.IndexError(e) from e
        n = max(start or 0, stop or 0) - len(self._cache) + 1
        if n > 0:
            self._cache.extend(itertools.islice(self._iterable, n))
        try:
            return self._cache[idx]
        except IndexError as e:
            raise self.IndexError(e) from e

    def __bool__(self):
        try:
            self[-1] if self._reversed else self[0]
        except self.IndexError:
            return False
        return True

    def __len__(self):
        self._exhaust()
        return len(self._cache)

    def __reversed__(self):
        return type(self)(self._iterable, reverse=not self._reversed, _cache=self._cache)

    def __copy__(self):
        return type(self)(self._iterable, reverse=self._reversed, _cache=self._cache)

    def __repr__(self):
        # repr and str should mimic a list. So we exhaust the iterable
        return repr(self.exhaust())

    def __str__(self):
        return repr(self.exhaust())
