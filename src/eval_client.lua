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
local FITNESS_HEADER = "FITNESS:"
local LOG_HEADER = "LOG:"
local LOAD_SLOT = 2  -- the emulator savestate slot to load
local MAX_COINS = 50000
local MAX_GAMES = 255
local REQUEST_MODE = "REQUEST_MODE"
local MODE_TRAIN = "MODE_TRAIN"
local MODE_EVAL = "MODE_EVAL"

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

local function str_to_table(str)
    local t = {}
    for v in string.gmatch( str, "([%w%d%.]+)") do
       t[#t+1] = v
    end
    return t
end

local function log(msg)
	print(msg)
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

local function sort_actions(weights)
    return sort_by_values(weights, function(a, b) return a > b end)
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
	log("Randomizing seed: "..rng)
	mainmemory.write_u32_le(0x10F6CC, rng)
end

local function debug_seed(_seed)
	log("Debugging seed: ".._seed)
	mainmemory.write_u32_le(0x10F6CC, _seed)
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
	log("Sending screenshot & game states...")
	comm.socketServerSend(VISIBLE_STATE_HEADER..serialize_table(visible_state))
	comm.socketServerResponse()
	comm.socketServerSend(HIDDEN_STATE_HEADER..serialize_table(hidden_state))
	comm.socketServerResponse()
	local response = comm.socketServerScreenShotResponse()
	return response
end

local function send_game_fitness()
	local fitness = read_collected_coins()
	log("fitness score: "..fitness)
	comm.socketServerSend(FITNESS_HEADER..fitness)
	-- comm.socketServerResponse()
end

local function count_remaining_tiles(visible_state, hidden_state)
	local count = 0
	for i = 0, 24, 1 do
		local tile_visual = visible_state.tiles[""..i]
		local tile_hidden = hidden_state.tiles[""..i]
		-- count hidden 2/3 coin tiles
		if tile_visual == 0x0 and (tile_hidden == 0x2 or tile_hidden == 0x3) then
			count = count + 1
		end
	end
	return count
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

local function auto_level()
	log("Starting auto level...")
	local visible_state = init_visibility_state()
	local hidden_state = read_hidden_state()
	local tiles = read_tiles()
	local sorted_tiles = sort_by_values(tiles, function(a, b) return a < b end)

	-- screenshot
	send_game_states(visible_state, hidden_state)

	for _, idx in pairs(sorted_tiles) do
		local item = tiles[idx]
		if item ~= 4 then
			log(idx, item)
			select_tile(tonumber(idx))
			visible_state.tiles[idx] = item
			advance_dialogue_state()
			-- screenshot
			send_game_states(visible_state, hidden_state)
		else
			log(idx, item)
			visible_state.tiles[idx] = item
			-- no screenshot
		end
	end
	log("Auto level cleared.")
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

local function manual_level()
	log("Starting manual level...")
	local visible_state = init_visibility_state()
	local hidden_state = read_hidden_state()

	local success = true
	local remaining_count = count_remaining_tiles(visible_state, hidden_state)
	while success and remaining_count > 0 do
		log("Remaining tile count: "..remaining_count)
		local decision_map = str_to_table(send_game_states(visible_state, hidden_state))
		local action_weights = sort_actions({table.unpack(decision_map, 1, 25)})
		-- log(action_weights)

		local i = 1
		local decision = tostring(action_weights[i] - 1)
		while visible_state.tiles[decision] ~= 0x0 do
			i = i + 1
			decision = tostring(action_weights[i] - 1)
			-- print(decision.." "..visible_state.tiles[decision])
		end

		local truth_value = hidden_state.tiles[decision]
		log("Selecting tile("..decision.."): truth_value="..truth_value)
		select_tile(tonumber(decision))
		visible_state.tiles[decision] = truth_value
		advance_dialogue_state()
		success = truth_value ~= 0x4
		remaining_count = count_remaining_tiles(visible_state, hidden_state)
	end

	log("Manual level success: "..tostring(success))
	advance_frames({}, 200)
	advance_dialogue_state()
	while not (in_menu_dialogue()) do
		advance_frames({["A"] = "True"}, 1)
		advance_frames({}, 5)
	end
	advance_dialogue_state()

	return success
end


-- ####################################
-- ####         GAME LOOP          ####
-- ####################################
function GameLoop(eval_mode)
	local game_idx = 0
	while game_idx < MAX_GAMES do
		log("Beginning Game("..game_idx..")...")

		-- load save state
		log("Loading save slot "..LOAD_SLOT.."...")
		savestate.loadslot(LOAD_SLOT)
		-- randomize_seed()
		debug_seed(game_idx)
		advance_lobby_state()

		-- loop until a round is lost or max currency is reached
		local success = true
		while read_collected_coins() < MAX_COINS and success do
			advance_dialogue_state()
			if eval_mode == MODE_TRAIN then
				auto_level()
			else
				success = manual_level()
			end
			log("Collected coins: "..read_collected_coins())
			advance_frames({}, 200)
		end

		-- end game loop
		log("Finished game loop.")
		send_game_fitness()
		game_idx = game_idx + 1
	end
end


-- ####################################
-- ####           MAIN             ####
-- ####################################
print("Is client connected to socket server?")
print(comm.socketServerIsConnected())
print(comm.socketServerGetInfo())

-- wait for server to be ready
comm.socketServerSend(READY_STATE)
local server_state = comm.socketServerResponse()
log("Server State: "..server_state)
if server_state == READY_STATE then
	-- request eval mode
	comm.socketServerSend(REQUEST_MODE)
	local eval_mode = comm.socketServerResponse()
	log("Evaluation mode: "..eval_mode)
	-- start game loop
	GameLoop(eval_mode)
elseif server_state == FINISHED_STATE then
	-- Close emulator
	client.exit()
end