# .vsrg File Format V1

## Notation

```
u8 / u16 / u32    unsigned integer of that bit width
T[n]              array of n elements of type T
T[]               array of unknown size (total size of all elements can)
str               string
```

All integers are little-endian.

Every section's size is calculable from fields declared before it and stored in the corresponding meta positions such that the implementer can calculate each section's size without parsing for headers.

Each `size of .X` counts every byte in that section. For `.meta` that includes its own version field, reserved block, and size table, measured from the first byte after the starting delimiter.

In string arrays, null terminators are reserved. Every single string in the array is terminated, including the final string. Strings are encoded in UTF-8 to avoid locale translation problems. This means locale conversion must be done by the implementation!

## Starting Delimiter

```
62 65 61 74 73 6F 75 70    ("beatsoup" in ASCII)
```

## Sections (Must be in this order)

### `.meta`

```
u16       .vsrg file format version (set to 1 for this version)
u8[62]    reserved (bytes 3-64)
u32       size of .meta in bytes
u32       size of .resources in bytes
u32       size of .chart in bytes
u32       size of .grid in bytes
u32       size of .note in bytes
str[4]    song title, song artist, tags, song file
```

### `.resources`

```
str[]     valid file paths
```

An array of strings of **valid** file paths

- Can be indexed like `resources[n]`, where `n` corresponds to the resource id.
- File resolution should be handled by an higher level implementation

### `.chart`

```
u8        column count
u8        note type
u8        grid parameter count
u8        note parameter count
```

### `.grid`

How we quantize the grid into ticks. For each row under `.grid`

- `tick` corresponds to the tick val to create the change after integrating all functions before it.
- `function number` maps to a time function that takes in the current time and spits out a tick value
  - `0-127` reserved
  - `128+` implementer defined
- Each parameter is a `u32`, parameters are implementer defined

```
u32 tick | u32 function number | u32[grid parameter count] parameters

row_size = num grid * (8 + 4 * grid parameter count)
```

### `.notes`

- `column idx` - which column for this note
- `note type id` - type of note, is it a roll, ln, tap, mine etc, whatever the development intends
  - `0-127` reserved
  - `128-255` implementer defined
- `start tick` - when the note starts
- `end tick` - when the note ends, could be the same as as start if needed
- Each parameter is a `u32`, parameters are implementer defined

```
u8 column idx | u8 note type id | u32 start tick | u32 end tick | u32[note parameter count] parameters

row_size = num notes * (10 + 4 * note parameter count)
```

## Ending Delimiter

```
DE AD BE A7
```
