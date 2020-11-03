var annData = {
    entity_types: [
          {
            type   : 'Person',
            labels : ['Person'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
          },
          {
            type   : 'Rank',
            labels : ['Rank'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
          },
          {
            type   : 'Organization',
            labels : ['Organization'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
          },
          {
            type   : 'Title',
            labels : ['Title'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
          },
          {
            type   : 'Role',
            labels : ['Role'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
          },
          {
            type   : 'Title_Role',
            labels : ['Title_Role'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
          },
          {
            type   : 'Location',
            labels : ['Location'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
          },
          {
            type   : 'Misclass',
            labels : ['Misclass'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
          }
        ]
};

var cur_SDP_data = {
    text     : "Shortest Dependency Path",
    entities : [],
    relations : [],
};

var cur_NN_data = {
    text     : "Neural Network",
    entities : [],
    relations : [],
};

var webFontURLs = "js/";
var doc_size_limit = 1 * 1024;
var kge_time_has_alerted = false;
var input_buffer;
var oid;

function input_too_large_msg(filesize) {
  var size_limit_in_kb = doc_size_limit / 1024;
  return "The document size is: " + filesize.toString() + " Bytes\n" +
        "You can only upload one text file less than " + size_limit_in_kb.toString() + "KB!\n" +
        "Download the algorithm at \n" +
        "Run it locally with no file size limit and batch input support!";
}

function check_loading() {
  setTimeout(function() {
    // Check if brat has loaded
    if ($("#SDP_graph").find('.text').length == 1 &&
        $("#NN_graph").find('.text').length == 1)
    {
      $(".brat_loading").remove();
      $("#load_warning").remove();
      $(".after_loading").css("visibility", "visible");
      end_processing();
      return;
    } else {
      check_loading();
    }
  }, 100);
}

function start_processing() {
  $("#processing_kge").css("display", "block");
  $("#extract_btn").css("display", "none");
  $("#input_form").css("display", "none");
  if (!kge_time_has_alerted) {
    alert("Extracting a Knowledge Graph could take up to several minutes, depending on the size of the input");
    kge_time_has_alerted = true;
  }
}

function end_processing() {
  $("#processing_kge").css("display", "none");
  $("#extract_btn").css("display", "inline-block");
  $("#input_form").css("display", "block");
}

$(document).ready(function(e){

    $.ajax({
        type: "GET",
        url: "/get_size_limit",
        success: function(res){
          console.log(res["size_limit"]);
          doc_size_limit = res["size_limit"];
          $('#maximum').html("/ " + doc_size_limit.toString());
          $("#input_txtbox").attr("maxlength", doc_size_limit);
        },
        error: function (e) {
          alert(e.responseText);
        }
    });

    var dispatcher_SDP;
    head.ready(function() {
        dispatcher_SDP = Util.embed('SDP_graph', annData, cur_SDP_data, webFontURLs);
    });

    var dispatcher_NN;
    head.ready(function() {
        dispatcher_NN = Util.embed('NN_graph', annData, cur_NN_data, webFontURLs);
    });

    $( "#SDP_graph" ).change(function() {
        alert( "Handler for .change() called." );
    });

    check_loading();

    // Submit form data via Ajax
    $("#input_form").on('submit', function(e){
        if ($("#file_selected").val() === "") {
          alert("You have not selected a file!");
          return false;
        } else {
          start_processing();

          e.preventDefault();
          var formData = new FormData(this);
          $.ajax({
              type: "POST",
              url: "/upload",
              data: formData,
              processData: false,
              contentType: false,
              success: function(res){
                  end_processing();
                  cur_SDP_data = res['json_SDP'];
                  cur_NN_data = res['json_NN'];
                  dispatcher_SDP.post('requestRenderData', [cur_SDP_data]);
                  dispatcher_NN.post('requestRenderData', [cur_NN_data]);
                  input_buffer = res['input_buffer'];
                  oid = res['oid'];
              },
              error: function (e) {
                alert(e.responseText);
              }
          });
        }
    });

    $("#extract_btn").on('click', function(e){
        var txtbox = $('#input_txtbox').val();
        if (txtbox === ""){
          alert("Text cannot be empty!");
        } else if (txtbox.length > doc_size_limit) {
          alert(input_too_large_msg(txtbox.length));
        } else {
          start_processing();
          $.ajax({
              type: "POST",
              url: "/upload_txt",
              data: {"txtbox" : txtbox},
              dataType: "json",
              success: function(res){
                  end_processing();
                  cur_SDP_data = res['json_SDP'];
                  cur_NN_data = res['json_NN'];
                  dispatcher_SDP.post('requestRenderData', [cur_SDP_data]);
                  dispatcher_NN.post('requestRenderData', [cur_NN_data]);
                  input_buffer = res['input_buffer'];
                  oid = res['oid'];
              },
              error: function (e) {
                alert(e.responseText);
              }
          });
        }
    });

    $('#file_selected').on('change', function() {
        if (this.files[0].size > doc_size_limit) {
            alert(input_too_large_msg(this.files[0].size));
            document.getElementById('file_selected').value = '';
            $("#file_selection_label").html("Select a text file...");
        } else {
            var input_path = $("#file_selected").val();
            var input_name = input_path.replace(/^.*[\\\/]/, '')
            $("#file_selection_label").html(input_name);
        }
    });

    $('textarea').keyup(function() {
      // Reference: https://codepen.io/zabielski/pen/gPPywv
      var characterCount = $(this).val().length,
          current = $('#current'),
          maximum = $('#maximum'),
          theCount = $('#the-count');

      current.text(characterCount);
      warning_threshold = Math.floor(doc_size_limit * 0.9);

      if (characterCount == 0) {
        maximum.css('color', 'black');
        current.css('color', 'black');
      }
      if ( 0 < characterCount && characterCount <  warning_threshold) {
        maximum.css('color', '#32CD32');
        current.css('color', '#32CD32');
      }
      if (characterCount >= warning_threshold) {
        maximum.css('color', '#DC143C');
        current.css('color', '#DC143C');
        theCount.css('font-weight','bold');
      } else {
        theCount.css('font-weight','normal');
      }
    });

    $( window ).resize(function() {
      dispatcher_SDP.post('requestRenderData', [cur_SDP_data]);
      dispatcher_NN.post('requestRenderData', [cur_NN_data]);
    });

    // ================= Examples =================
    $("#ex1").click(function(){
      $.ajax({
          type: "POST",
          url: "/get_example",
          data: {"ex_num" : 1},
          dataType: "json",
          success: function(res){
              end_processing();
              // cur_SDP_data = res['json_SDP'];
              // cur_NN_data = res['json_NN'];
              // dispatcher_SDP.post('requestRenderData', [cur_SDP_data]);
              // dispatcher_NN.post('requestRenderData', [cur_NN_data]);
              var text = res['json_SDP']['text'];
              $("#input_txtbox").val(text);
          },
          error: function (e) {
            alert(e.responseText);
          }
      });
    });

    $("#ex2").click(function(){
      $.ajax({
          type: "POST",
          url: "/get_example",
          data: {"ex_num" : 2},
          dataType: "json",
          success: function(res){
              end_processing();
              // cur_SDP_data = res['json_SDP'];
              // cur_NN_data = res['json_NN'];
              // dispatcher_SDP.post('requestRenderData', [cur_SDP_data]);
              // dispatcher_NN.post('requestRenderData', [cur_NN_data]);
              var text = res['json_SDP']['text'];
              $("#input_txtbox").val(text);
          },
          error: function (e) {
            alert(e.responseText);
          }
      });
    });

    $("#ex3").click(function(){
      $.ajax({
          type: "POST",
          url: "/get_example",
          data: {"ex_num" : 3},
          dataType: "json",
          success: function(res){
              end_processing();
              // cur_SDP_data = res['json_SDP'];
              // cur_NN_data = res['json_NN'];
              // dispatcher_SDP.post('requestRenderData', [cur_SDP_data]);
              // dispatcher_NN.post('requestRenderData', [cur_NN_data]);
              var text = res['json_SDP']['text'];
              $("#input_txtbox").val(text);
          },
          error: function (e) {
            alert(e.responseText);
          }
      });
    });

    $(".dropdown-item").click(function() {
      $("#home").css("display", "block");
      $("#about").css("display", "none");
    });

    $(".kge_title").click(function() {
      $("#home").css("display", "block");
      $("#about").css("display", "none");
    });

    $("#about_nav").click(function() {
      $("#home").css("display", "none");
      $("#about").css("display", "block");
    });

    // $("#download_sdp_ann").click(function(){
    //   $.ajax({
    //       type: "POST",
    //       url: "/download",
    //       data: {"input_buffer" : input_buffer,
    //               "oid" : oid,
    //               "method" : "SDP",
    //               "txt_ann" : "ann"},
    //       dataType: "json",
    //       success: function(res){
    //         console.log("Download");
    //       },
    //       error: function (e) {
    //         alert(e.responseText);
    //       }
    //   });
    // });

    $('#download_sdp_ann').click(function(e) {
        e.preventDefault();
        console.log(input_buffer);
        var file_url = "download/?file_path=" + input_buffer + '/SDP/' + oid + ".ann";
        console.log(file_url);
        window.open(file_url);
    });

    $('#download_nn_ann').click(function(e) {
        e.preventDefault();
        console.log(input_buffer);
        var file_url = "download/?file_path=" + input_buffer + '/NN/' + oid + ".ann";
        console.log(file_url);
        window.open(file_url);
    });

    $('#download_txt').click(function(e) {
        e.preventDefault();
        console.log(input_buffer);
        var file_url = "download/?file_path=" + input_buffer + '/SDP/' + oid + ".txt";
        console.log(file_url);
        window.open(file_url);
    });

});
