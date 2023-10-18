// Persistent network connection that will be used to transmit real-time data
var socket = io();
let sleep = ms => {  
    return new Promise(resolve => setTimeout(resolve, ms));  
    };  
/* * * * * * * * * * * * * * * * 
 * Button click event handlers *
 * * * * * * * * * * * * * * * */

$(function() {
    $('#create').click(function () { /* Function that creates the tutorial level*/
        params = arrToJSON($('form').serializeArray());
        params.layouts = [params.layout]
        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };
	 /*In tutorial level we should not pretend to be playing with a human - so you should leave this way - no sleep seconds*/   
	
        socket.emit("create", data);
        
	$('#waiting').show();
       
        
        
        $('#join').hide();
        $('#join').attr("disabled", true);
        
        $('#create').hide();
        $('#create').attr("disabled", true)

        $('#saveid').hide();
        $('#tutorial_code').hide();
        $('#one_code').hide();
        $('#two_code').hide();

        $("#instructions").hide();
        $('#tutorial').hide();

        $('#help_tutorial').show();
        $('#help_tutorial').attr("disabled", false);
    });
});

$(function() {
    $('#again').click(function () {
        params = arrToJSON($('form').serializeArray());
        params.layouts = [params.layout]
        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };
        socket.emit("create", data);
        console.log(data)
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create').hide();
        $('#create').attr("disabled", true)
        $('#tutorial_code').hide();
        $('#one_code').hide();
        $('#two_code').hide();

        $('#again').hide();
        $('#again').attr("disabled", true)
        $("#instructions").hide();
        $('#tutorial').hide();
        $('#saveid').hide();

        $('#help_tutorial').show();
        $('#help_tutorial').attr("disabled", false);
    });
});

$(function() {
    $('#help_tutorial').click(function () {
        socket.emit('leave', {});

        params = arrToJSON($('form').serializeArray());
        params.layouts = [params.layout]
        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };
        socket.emit("create", data);
        console.log(data)
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create').hide();
        $('#create').attr("disabled", true)
        $('#tutorial_code').hide();
        $('#one_code').hide();
        $('#two_code').hide();

        $('#again').hide();
        $('#again').attr("disabled", true)

        $('#help_tutorial').show();
        $('#help_tutorial').attr("disabled", false);

        $("#instructions").hide();
        $('#tutorial').hide();
        $('#saveid').hide();
    });
});

$(function() {
    $('#join').click(function() {
        socket.emit("join", {});
        $('#join').attr("disabled", true);
        $('#create').attr("disabled", true);
    });
});

$(function() {
    $('#leave').click(function() {
        socket.emit('leave', {});
        $('#leave').attr("disabled", true);
    });
});


$(function() {
    $('#create_one').click(function () {
        params = arrToJSON_one($('form').serializeArray());
        params.layouts = [params.layout]
        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };
        //Uncoment the following 6 lines of code for human_player_bluff condition
        var min = 5000;
        var max = 10000;
        var randomNumber = Math.floor(Math.random() * (max - min + 1)) + min;
        sleep(randomNumber).then(() => {  
            socket.emit("create", data);
        });
       
	//Coment next line if you are in human_player_buff condition    
        //socket.emit("create", data);
       
	console.log(data)
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create_one').hide();
        $('#create_one').attr("disabled", true)
        $('#tutorial_code').hide();
        $('#one_code').hide();
        $('#two_code').hide();

        $('#again').hide();
        $('#again').attr("disabled", true)

        $('#help_tutorial').hide();
        $('#help_tutorial').attr("disabled", true)
        

        $("#leave").hide();
        $('#leave').attr("disabled", true)
        $("#instructions").hide();
        $('#tutorial').hide();
        $('#saveid').hide();

        $('#help_one').show();
        $('#help_one').attr("disabled", false);
    });
});


$(function() {
    $('#help_one').click(function () {
        socket.emit('leave', {});

        params = arrToJSON_one($('form').serializeArray());
        params.layouts = [params.layout]
        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };
        socket.emit("create", data);
        console.log(data)
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create_one').hide();
        $('#create_one').attr("disabled", true)
        $('#tutorial_code').hide();
        $('#one_code').hide();
        $('#two_code').hide();

        $('#again').hide();
        $('#again').attr("disabled", true)

        $('#help_one').show();
        $('#help_one').attr("disabled", false);

        $("#instructions").hide();
        $('#tutorial').hide();
        $('#saveid').hide();
    });
});

$(function() {
    $('#create_two').click(function () {
        params = arrToJSON_two($('form').serializeArray());
        params.layouts = [params.layout]
        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };

	//Uncoment the following 6 lines of code for human_player_bluff condition    
       var min = 5000;
        var max = 10000;
        var randomNumber = Math.floor(Math.random() * (max - min + 1)) + min;
        sleep(randomNumber).then(() => {  
            socket.emit("create", data);
        });  
	
	//Coment next line if you are in human_player_buff condition    
        //socket.emit("create", data);
       
        console.log(data)
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create_two').hide();
        $('#create_two').attr("disabled", true)
        $('#again').hide();
        $('#again').attr("disabled", true)
        $("#leave").hide();
        $('#leave').attr("disabled", true)
        $("#instructions").hide();
        $('#tutorial_code').hide();
        $('#one_code').hide();
        $('#two_code').hide();
        $('#tutorial').hide();
        $('#saveid').hide();

        $('#help_two').show();
        $('#help_two').attr("disabled", false);

        $('#help_one').hide();
        $('#help_one').attr("disabled", true)

        


    });
});

$(function() {
    $('#help_two').click(function () {
        socket.emit('leave', {});

        params = arrToJSON_two($('form').serializeArray());
        params.layouts = [params.layout]
        data = {
            "params" : params,
            "game_name" : "overcooked",
            "create_if_not_found" : false
        };
        socket.emit("create", data);
        console.log(data)
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create_one').hide();
        $('#create_one').attr("disabled", true)
        $('#tutorial_code').hide();
        $('#one_code').hide();
        $('#two_code').hide();

        $('#again').hide();
        $('#again').attr("disabled", true)

        $('#help_two').show();
        $('#help_two').attr("disabled", false);

        $("#instructions").hide();
        $('#tutorial').hide();
        $('#saveid').hide();
    });
});






/* * * * * * * * * * * * * 
 * Socket event handlers *
 * * * * * * * * * * * * */

window.intervalID = -1;
window.spectating = true;

socket.on('waiting', function(data) {
    // Show game lobby
    $('#error-exit').hide();
    $('#waiting').hide();
    $('#game-over').hide();
    $('#instructions').hide();
    $('#tutorial').hide();
    $("#overcooked").empty();
    $('#lobby').show();
    $('#join').hide();
    $('#join').attr("disabled", true)
    $('#tutorial_code').hide();
    $('#one_code').hide();
    $('#two_code').hide();
    $('#saveid').hide();
    $('#create').hide();
    $('#create').attr("disabled", true)
    $('#leave').show();
    $('#leave').attr("disabled", false);
    if (!data.in_game) {
        // Begin pinging to join if not currently in a game
        if (window.intervalID === -1) {
            window.intervalID = setInterval(function() {
                socket.emit('join', {});
            }, 1000);
        }
    }
});

socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#lobby').hide();
    $("#instructions").show();
    $('#tutorial').show();
    $('#waiting').hide();
    $('#join').show();
    $('#join').attr("disabled", false);
    $('#create').show();
    $('#tutorial_code').hide();
    $('#one_code').hide();
    $('#two_code').hide();
    $('#create').attr("disabled", false);
    $('#overcooked').append(`<h4>Sorry, game creation code failed with error: ${JSON.stringify(err)}</>`);
});

socket.on('start_game', function(data) {
    // Hide game-over and lobby, show game title header
    if (window.intervalID !== -1) {
        clearInterval(window.intervalID);
        window.intervalID = -1;
    }
    graphics_config = {
        container_id : "overcooked",
        start_info : data.start_info
    };

    window.spectating = data.spectating;
    $('#error-exit').hide();
    $("#overcooked").empty();
    $('#game-over').hide();
    $('#lobby').hide();
    $('#waiting').hide();
    $('#join').hide();
    $('#join').attr("disabled", true);
    $('#create').hide();
    $('#create').attr("disabled", true);
    $('#create_one').hide();
    $('#create_one').attr("disabled", true);
    $('#create_two').hide(); 
    $('#create_two').attr("disabled", true);
    $("#instructions").hide();
    $('#tutorial').hide();
    $('#tutorial_code').hide();
    $('#one_code').hide();
    $('#two_code').hide();
    //$('#leave').show();
    //$('#leave').attr("disabled", false)
    $('#game-title').show();
    
    if (!window.spectating) {
        enable_key_listener();
    }
    
    graphics_start(graphics_config);
});

socket.on('reset_game', function(data) {
    graphics_end();
    if (!window.spectating) {
        disable_key_listener();
    }
    
    $("#overcooked").empty();
    $("#reset-game").show();
    setTimeout(function() {
        $("reset-game").hide();
        graphics_config = {
            container_id : "overcooked",
            start_info : data.state
        };
        if (!window.spectating) {
            enable_key_listener();
        }
        graphics_start(graphics_config);
    }, data.timeout);
});

socket.on('state_pong', function(data) {
    // Draw state update
    drawState(data['state']);
});

socket.on('end_game', function(data) {
    // Hide game data and display game-over html
    graphics_end();
    if (!window.spectating) {
        disable_key_listener();
    }
    $('#game-title').hide();
    $('#game-over').show();
    $('#game-over').attr("disabled", false);
    $("#join").show();
    $('#join').attr("disabled", false);
    console.log(data)
    console.log(data["data"]['layout'])
    if (data["data"]['layout'] === 'cramped_room_tutorial'){
        $("#again").show();
        $('#again').attr("disabled", false)
        $('#tutorial_code').show();
        $("#create_one").show();
        $('#create_one').attr("disabled", false)
        $('#help_tutorial').hide();
        $('#help_tutorial').attr("disabled", true)
        
        
    }
    if (data["data"]['layout'] === 'level_one'){
        
        $("#create_two").show();
        $('#create_two').attr("disabled", false)
        
        $('#one_code').show();
        $('#help_one').hide();
        $('#help_one').attr("disabled", true)
        
    }   
    if(data["data"]['layout'] === 'level_two') {
        $('#two_code').show();
        $('#help_two').hide();
        $('#help_two').attr("disabled", true)
    } 
   

    

    $('#create').hide();
    $('#create').attr("disabled", false)


    //$("#overcooked").hide();
    //$('#overcooked').attr("disabled", true)
    //$("#instructions").show();
    //$('#tutorial').show();
    $("#leave").hide();
    $('#leave').attr("disabled", true)
    
    // Game ended unexpectedly
    if (data.status === 'inactive') {
        $('#error-exit').show();
    }
});

socket.on('end_lobby', function() {
    // Hide lobby
    $('#lobby').hide();
    $("#join").show();
    $('#join').attr("disabled", false);
    $("#create").show();
    $('#create').attr("disabled", false)
    $("#leave").hide();
    $('#leave').attr("disabled", true)
    $("#instructions").show();
    $('#tutorial').show();

    // Stop trying to join
    clearInterval(window.intervalID);
    window.intervalID = -1;
})


/* * * * * * * * * * * * * * 
 * Game Key Event Listener *
 * * * * * * * * * * * * * */

function enable_key_listener() {
    $(document).on('keydown', function(e) {
        let action = 'STAY'
        switch (e.which) {
            case 37: // left
                action = 'LEFT';
                break;

            case 38: // up
                action = 'UP';
                break;

            case 39: // right
                action = 'RIGHT';
                break;

            case 40: // down
                action = 'DOWN';
                break;

            case 32: //space
                action = 'SPACE';
                break;

            default: // exit this handler for other keys
                return; 
        }
        e.preventDefault();
        socket.emit('action', { 'action' : action });
    });
};

function disable_key_listener() {
    $(document).off('keydown');
};


/* * * * * * * * * * *
 * Utility Functions *
 * * * * * * * * * * */

var arrToJSON = function(arr) {
    let retval = {}
    console.log("O meu print")
    for (let i = 0; i < arr.length; i++) {
        elem = arr[i];
        key = elem['name'];
        value = elem['value'];
        retval[key] = value;
        console.log(retval[key])
    }
    

    return retval;
};

var arrToJSON_one = function(arr) {
    let retval = {}
    console.log("one print")
    for (let i = 0; i < arr.length; i++) {
        elem = arr[i];
        if (elem['value'] == "cramped_room_tutorial"){
            value= "level_one";
        }
        else {
            value = elem['value'];
        }
        key = elem['name'];
        retval[key] = value;
        console.log(retval[key])
    }
    

    return retval;
};

var arrToJSON_two = function(arr) {
    let retval = {}
    console.log("two print")
    for (let i = 0; i < arr.length; i++) {
        elem = arr[i];
        if (elem['value'] == "cramped_room_tutorial"){
            value= "level_two";
        }
        else {
            value = elem['value'];
        }
        key = elem['name'];
        retval[key] = value;
        console.log(retval[key])
    }
    

    return retval;
};

