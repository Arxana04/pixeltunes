export type SpotifyLinkType = 'track' | 'album' | 'playlist';

export function parseSpotifyLink(input: string) {
  const value = input.trim();
  let match = /^spotify:(track|album|playlist):([A-Za-z0-9]{22})$/.exec(value);

  if (!match) {
    let url: URL;
    try { url = new URL(value); } catch { return null; }
    if (url.origin !== 'https://open.spotify.com' || url.username || url.password) return null;
    match = /^\/(?:intl-[a-z-]+\/)?(?:embed\/)?(track|album|playlist)\/([A-Za-z0-9]{22})\/?$/i.exec(url.pathname);
  }

  if (!match) return null;
  const type = match[1].toLowerCase() as SpotifyLinkType;
  const id = match[2];
  return {
    type,
    id,
    uri: `spotify:${type}:${id}`,
    url: `https://open.spotify.com/${type}/${id}`,
  };
}

export function getSpotifyItemUrl(uri: string) {
  const episode = /^spotify:episode:([A-Za-z0-9]{22})$/.exec(uri);
  return episode ? `https://open.spotify.com/episode/${episode[1]}` : parseSpotifyLink(uri)?.url;
}

const SPOTIFY_URL_PREFIX = {
  track: 'https://open.spotify.com/track/',
  album: 'https://open.spotify.com/album/',
  playlist: 'https://open.spotify.com/playlist/',
} as const;
const ID_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

/**
 * Build an open.spotify.com URL for a parsed link without carrying user text
 * into the result: the prefix comes from a constant table and every ID
 * character is re-read from a fixed alphabet. Returns undefined if anything
 * falls outside those sets.
 */
export function buildSafeSpotifyUrl(link: { type: SpotifyLinkType; id: string }): string | undefined {
  const prefix = SPOTIFY_URL_PREFIX[link.type as keyof typeof SPOTIFY_URL_PREFIX];
  if (!prefix || link.id.length !== 22) return undefined;
  let id = '';
  for (const ch of link.id) {
    const index = ID_ALPHABET.indexOf(ch);
    if (index < 0) return undefined;
    id += ID_ALPHABET.charAt(index);
  }
  return prefix + id;
}
