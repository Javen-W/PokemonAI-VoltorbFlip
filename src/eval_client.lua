--[[
References:
	- https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_data_structure_(Generation_IV)#Encrypted_bytes_2
	- https://bulbapedia.bulbagarden.net/wiki/Save_data_structure_(Generation_IV)
	- https://projectpokemon.org/home/docs/gen-4/platinum-save-structure-r81/
	- https://projectpokemon.org/docs/gen-4/pkm-structure-r65/
	- https://tasvideos.org/UserFiles/Info/45193606014900979
--]]

--[[
	### CONSTANTS ###
--]]
local READY_STATE = "READY"
local FINISHED_STATE = "FINISHED"
local VISIBLE_STATE_HEADER = "VISIBLE_STATE:"
local HIDDEN_STATE_HEADER = "HIDDEN_STATE:"
local LOG_HEADER = "LOG:"
local LOAD_SLOT = 2  -- the emulator savestate slot to load
local MAX_COINS = 50000

-- used in PRNG state calculation
local function mult32(a, b)
	local c = a >> 16
	local d = a % 0x10000
	local e = b >> 16
	local f = b % 0x10000
	local g = (c * f + d * e) % 0x10000
	local h = d * f
	local i = g * 0x10000 + h
	return i
end

local function log(msg)
    comm.socketServerSend(LOG_HEADER..tostring(msg))
end

function table.shallow_copy(t)
	local t2 = {}
	for k,v in pairs(t) do
		t2[k] = v
	end
	return t2
end

local function serialize_table(tabl, indent, nl)
    nl = nl or string.char(10) -- newline
    indent = indent and (indent.."  ") or ""
    local str = ''
    str = str .. indent.."{"
    for key, value in pairs (tabl) do
        local pr = (type(key)=="string") and ('"'..key..'":') or ""
        if type (value) == "table" then
            str = str..nl..pr..serialize_table(value, indent)..','
        elseif type (value) == "string" then
            str = str..nl..indent..pr..'"'..tostring(value)..'",'
        else
            str = str..nl..indent..pr..tostring(value)..','
        end
    end
    str = str:sub(1, #str-1) -- remove last symbol
    str = str..nl..indent.."}"
    return str
end

local function sort_by_values(tbl, sort_function)
    local keys = {}
    for key in pairs(tbl) do
        table.insert(keys, key)
    end
    table.sort(keys, function(a, b)
        return sort_function(tbl[a], tbl[b]) end
    )
    return keys
end

local function decrypt(seed, addr, words)
	-- decrypt pokemon data bytes
	local X = { seed }
	local D = {}
	for n = 1, words+1, 1 do
		X[n+1] = mult32(X[n], 0x41C64E6D) + 0x6073
		D[n] = memory.read_u16_le(addr + ((n - 1) * 0x02))
		D[n] = D[n] ~ (X[n+1] >> 16)
		-- print(n, string.format("%X", D[n]))
	end
	return D
end

local function numberToBinary(x)
	local ret = ""
	while x~=1 and x~=0 do
		ret = tostring(x % 2)..ret
		x = math.modf(x / 2)
	end
	ret = tostring(x)..ret
	return ret
end

local function advance_frames(instruct, cnt)
    cnt = cnt or 1
    instruct = instruct or {}
    for i=0, cnt, 1 do
        emu.frameadvance()
        joypad.set(instruct)
    end
end

local function randomize_seed()
	math.randomseed(os.time())
	local rng = math.random(1, 250)
	print("Randomizing seed: "..rng)
	mainmemory.write_u32_le(0x10F6CC, rng)
end

local function read_tile(idx)
	local addr = 0x2E5EC4 + (idx * 0xC)
	return mainmemory.read_u32_le(addr)
end

local function read_tiles()
	local tiles_struct = {}
	for i = 0, 24, 1 do
		tiles_struct[""..i] = read_tile(i)
	end
	return tiles_struct
end

local function read_collected_coins()
	return mainmemory.read_u16_le(0x27C31C)
end

local function read_coin_counts()
	local coins = { }
	for i = 0, 9, 1 do
		local addr = 0x2E5FF0 + (i * 0x1)
		coins[""..i] = mainmemory.read_u8(addr)
	end
	return coins
end

local function read_bomb_counts()
	local bombs = { }
	for i = 0, 9, 1 do
		local addr = 0x2E5FFA + (i * 0x1)
		bombs[""..i] = mainmemory.read_u8(addr)
	end
	return bombs
end

local function read_dialogue_state()
	return mainmemory.read_u16_le(0x2C42B6)
end

local function in_game_dialogue()
	return read_dialogue_state() == 0xD002
end

local function in_menu_dialogue()
	return read_dialogue_state() == 0xD008
end

local function in_lobby_state()
	return mainmemory.read_u32_le(0x1D0DF0) ~= 0x7A
end

local function in_tile_selection()
	return read_dialogue_state() == 0x0000
end

local function advance_dialogue_state()
	while (in_menu_dialogue() or in_game_dialogue()) do
		advance_frames({["A"] = "True"}, 1)
		advance_frames({}, 5)
	end
end

local function advance_lobby_state()
	while not (in_menu_dialogue()) do
		advance_frames({["A"] = "True"}, 1)
		advance_frames({}, 5)
	end
	advance_dialogue_state()
end

local function read_cursor_index()
	return mainmemory.read_u8(0x2E5E61)
end

local function init_visibility_state()
	local visibility_state = {
		tiles = { },
		coins = read_coin_counts(),
		bombs = read_bomb_counts(),
	}
	for i = 0, 24, 1 do
		visibility_state.tiles[""..i] = 0x0
	end
	return visibility_state
end

local function read_hidden_state()
	local hidden_state = {
		tiles = read_tiles(),
		coins = read_coin_counts(),
		bombs = read_bomb_counts(),
	}
	return hidden_state
end

local function send_game_states(visible_state, hidden_state)
	advance_frames({}, 100) -- buffer while potential dialogue loads
	advance_dialogue_state()
	print("sending screenshot & game states...")
    comm.socketServerSend(VISIBLE_STATE_HEADER..serialize_table(visible_state))
	comm.socketServerResponse()
    comm.socketServerSend(HIDDEN_STATE_HEADER..serialize_table(hidden_state))
	comm.socketServerResponse()
	comm.socketServerScreenShotResponse()
	return true
end

local function select_tile(t_idx)
	if t_idx < 0 or t_idx >= 25 then
		return
	end
	-- advance to correct tile index
	local c_idx = read_cursor_index()
	while c_idx ~= t_idx do
		local c_row = math.floor(c_idx / 5)
		local c_col = c_idx % 5
		local t_row = math.floor(t_idx / 5)
		local t_col = t_idx % 5
		
		if c_row < t_row then
			advance_frames({["Down"] = "True"}, 1)
			advance_frames({}, 1)
		elseif c_row > t_row then
			advance_frames({["Up"] = "True"}, 1)
			advance_frames({}, 1)
		elseif c_col < t_col then
			advance_frames({["Right"] = "True"}, 1)
			advance_frames({}, 1)
		elseif c_col > t_col then
			advance_frames({["Left"] = "True"}, 1)
			advance_frames({}, 1)
		end
		c_idx = read_cursor_index()
	end
	-- select tile action
	advance_frames({["A"] = "True"}, 20)
	advance_frames({}, 1)
end

local function select_coin_tiles()
	print("Starting level.")
	local visible_state = init_visibility_state()
	local hidden_state = read_hidden_state()
	local tiles = read_tiles()
	local sorted_tiles = sort_by_values(tiles, function(a, b) return a < b end)

	-- screenshot
	send_game_states(visible_state, hidden_state)

	for _, idx in pairs(sorted_tiles) do
		local item = tiles[idx]
		if item ~= 4 then
			print(idx, item)
			select_tile(tonumber(idx))
			visible_state.tiles[idx] = item
			advance_dialogue_state()
			-- screenshot
			send_game_states(visible_state, hidden_state)
		else
			print(idx, item)
			visible_state.tiles[idx] = item
			-- no screenshot
		end
	end
	print("Level clear.")
	advance_frames({}, 200)
	advance_dialogue_state()
	-- screenshot
	send_game_states(visible_state, hidden_state)
	while not (in_menu_dialogue()) do
		advance_frames({["A"] = "True"}, 1)
		advance_frames({}, 5)
	end
	advance_dialogue_state()
end


-- ####################################
-- ####         GAME LOOP          ####
-- ####################################
function GameLoop()
	while true do
		log("Beginning game loop...")

		-- load save state
		log("Loading save slot "..LOAD_SLOT.."...")
		savestate.loadslot(LOAD_SLOT)
		randomize_seed()
		advance_lobby_state()

		-- loop until a round is lost or max currency is reached
		while read_collected_coins() < MAX_COINS do
			advance_dialogue_state()
			select_coin_tiles()
			print("Advancing to next level...")
			print("Collected coins: "..read_collected_coins())
			advance_frames({}, 200)
		end

		-- end game loop
		log("Finished game loop.")
	end
end


--GameLoop()
-- repeat game loop until evaluation server finishes
print("Is client connected to socket server?")
print(comm.socketServerIsConnected())
print(comm.socketServerGetInfo())

while true do
    comm.socketServerSend(READY_STATE)
    local server_state = comm.socketServerResponse()
    print("Server State: "..server_state)
    if server_state == READY_STATE then
        -- start game loop
    	GameLoop()
    elseif server_state == FINISHED_STATE then
        -- Close emulator
        client.exit()
    end
end