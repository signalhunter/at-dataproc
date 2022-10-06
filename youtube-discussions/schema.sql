CREATE TABLE youtube_discussions (
  channel_id String NOT NULL,
  comment_id String NOT NULL,
  author_id String NULL,
  author String NULL,
  timestamp DateTime NOT NULL,
  like_count UInt16 NOT NULL,
  favorited Boolean NOT NULL,
  text String NULL CODEC(ZSTD),
  profile_pic String NOT NULL CODEC(ZSTD)
)
ENGINE ReplacingMergeTree()
ORDER BY (channel_id, comment_id);
