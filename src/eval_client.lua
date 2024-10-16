--[[
References:
	- https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_data_structure_(Generation_IV)#Encrypted_bytes_2
	- https://bulbapedia.bulbagarden.net/wiki/Save_data_structure_(Generation_IV)
	- https://projectpokemon.org/home/docs/gen-4/platinum-save-structure-r81/
	- https://projectpokemon.org/docs/gen-4/pkm-structure-r65/
	- https://tasvideos.org/UserFiles/Info/45193606014900979
--]]


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
    comm.socketServerSend("LOG:"..tostring(msg))
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

local function randomize_seed()
    if ADD_RNG and is_outside() then
    	-- math.randomseed(os.time())
    	-- local rng = math.random(1, 250)
    	comm.socketServerSend("SEED")
        local rng = comm.socketServerResponse()
    	log("Randomizing seed: waiting "..rng.." frames...")
        advance_frames({}, rng)
    end
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
        ttl = ttl - 1
        refresh_gui()
    end
end

-- sends input state to server for evaluation
local function eval_state()
    read_inputstate()
    comm.socketServerSend("STATE"..serialize_table(input_state))  -- send state to eval server
    return str_to_table(comm.socketServerResponse())
end


local LOAD_SLOT = 2  -- the emulator save slot to load

-- ####################################
-- ####         GAME LOOP          ####
-- ####################################
function GameLoop()
    log("Beginning game loop...")

    -- initialize global vars

    -- load save state
    log("Loading save slot "..LOAD_SLOT.."...")
    savestate.loadslot(LOAD_SLOT)
    -- client.invisibleemulation(true)

    -- loop until a round is lost or TTL runs out
    while true do
        -- check game state
		
        -- is evaluation over?
		
        -- state advancement
		-- local decision = comm.socketServerScreenShotResponse()

        -- advance single frame
        advance_frames({}, 1)
    end

    -- end game loopx
    log("Finished game loop.")
    -- comm.socketServerSend("FITNESS:"..fitness)
    -- advance_frames({}, 250) -- buffer while server prepares
    -- return fitness
end


GameLoop()
--[[
-- repeat game loop until evaluation server finishes
print("Is client connected to socket server?")
print(comm.socketServerIsConnected())
print(comm.socketServerGetInfo())

while true do
    comm.socketServerSend("READY")
    local server_state = comm.socketServerResponse()
    print("Server State: "..server_state)
    if server_state == "READY" then
        -- start game loop
    	GameLoop()
    elseif server_state == "FINISHED" then
        -- Close emulator
        client.exit()
    end
end
--]]