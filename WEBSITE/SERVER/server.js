var express = require('express');
var app = express();
var path = require('path');
var multer  = require('multer');
var fs = require('fs');
const { exec } = require("child_process");
var bodyParser=require('body-parser');

app.use(bodyParser.urlencoded({
    extended: true
}));
app.use(bodyParser.json());

var txt_count = 0;
var doc_size_limit = 4 * 1024;
var UPLOAD_DIR = 'UPLOADs';
var TEXTBOX_DIR = "TEXTBOX";
var EXAMPLE_DIR = "EXAMPLES"
var SDP_DIR = "SDP";
var NN_DIR = "NN";
var upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: doc_size_limit }, // 1 MB
  fileFilter: function (req, file, cb) {
    // var filetypes = /zip|text/;
    var filetypes = /text/;
    if (filetypes.test(file.mimetype)) {
      return cb(null, true);
    }
    req.fileValidationError = "Error: File upload only supports the following filetypes - " + filetypes;
    return cb(null, false, new Error('goes wrong on the mimetype'));
  }
});

if (!String.prototype.format) {
  String.prototype.format = function() {
    var args = arguments;
    return this.replace(/{(\d+)}/g, function(match, number) {
      return typeof args[number] != 'undefined'
        ? args[number]
        : match
      ;
    });
  };
}

app.use('/', (req, res, next) => {
  console.log(req.method +  " request from " + req.connection.remoteAddress + " at " + req.url);
  next()
})

app.use('/', express.static(path.join(__dirname, '..')))

app.get('/get_size_limit', function(req, res) {
  res.json({"size_limit" : doc_size_limit});
})

app.post('/upload', upload.single('inputfile'), function (req, res, next) {
  // req.file is the `inputfile` file
  // req.body will hold the text fields, if there were any
  if(req.fileValidationError) {
    res.status(415)
    return res.end(req.fileValidationError);
  } else {
    // File ID, Original ID
    var fid = req.file.filename;
    var oid = req.file.originalname.slice(0, -4)
    var txt_filename = oid + ".txt";
    var input_buffer = UPLOAD_DIR + "/" + fid;
    var cwd = process.cwd();
    var command = "mv {0} {1} ; ".format(input_buffer, UPLOAD_DIR + "/" + txt_filename);
    command += "mkdir {0} ; ".format(input_buffer);
    command += "mv {0} {1} ; ".format(UPLOAD_DIR + "/" + txt_filename, input_buffer);
    command += "cd KGE ; ";
    command += "python pipeline.py {0} ; ".format(path.join(cwd, input_buffer));
    command += "cd RE/utils ; "
    command += "python ann_json.py {0} {1} {2} ; ".format(
        path.join(cwd, input_buffer, SDP_DIR, txt_filename),
        path.join(cwd, input_buffer, SDP_DIR, oid + '.ann'),
        path.join(cwd, input_buffer, SDP_DIR, oid + '.json') );
    command += "python ann_json.py {0} {1} {2} ; ".format(
        path.join(cwd, input_buffer, NN_DIR, txt_filename),
        path.join(cwd, input_buffer, NN_DIR, oid + '.ann'),
        path.join(cwd, input_buffer, NN_DIR, oid + '.json') );

    exec(command, (error, stdout, stderr) => {
        // if (error) {
        //     console.log(`error: ${error.message}`);
        //     return;
        // }
        // if (stderr) {
        //     console.log(`stderr: ${stderr}`);
        //     return;
        // }
        console.log('Annotations generated for file(s) under directory: ' + fid);

        var json_path_SDP = path.join(cwd, input_buffer, SDP_DIR, oid + '.json');
        var json_SDP = JSON.parse(fs.readFileSync(json_path_SDP, 'utf8'));

        var json_path_NN = path.join(cwd, input_buffer, NN_DIR, oid + '.json');
        var json_NN = JSON.parse(fs.readFileSync(json_path_NN, 'utf8'));

        var res_json = {
          "json_SDP" : json_SDP,
          "json_NN" : json_NN,
          "input_buffer" : input_buffer,
          "oid" : oid,
        };
        res.json(res_json);
    });

  }
})


app.post('/upload_txt', function(req, res) {
  var cur_id = txt_count.toString();
  txt_count++;

  var doc = req.body.txtbox;
  fs.writeFileSync(TEXTBOX_DIR + "/" + cur_id + '.txt', doc);

  // File ID, Original ID
  var fid = cur_id;
  var oid = cur_id;
  var txt_filename = oid + ".txt";
  var input_buffer = TEXTBOX_DIR + "/" + fid;
  var cwd = process.cwd();
  var command = "mkdir {0} ; ".format(input_buffer);
  command += "mv {0} {1} ; ".format(TEXTBOX_DIR + "/" + txt_filename, input_buffer);
  command += "cd KGE ; ";
  command += "python pipeline.py {0} ; ".format(path.join(cwd, input_buffer));
  command += "cd RE/utils ; "
  command += "python ann_json.py {0} {1} {2} ; ".format(
      path.join(cwd, input_buffer, SDP_DIR, txt_filename),
      path.join(cwd, input_buffer, SDP_DIR, oid + '.ann'),
      path.join(cwd, input_buffer, SDP_DIR, oid + '.json') );
  command += "python ann_json.py {0} {1} {2} ; ".format(
      path.join(cwd, input_buffer, NN_DIR, txt_filename),
      path.join(cwd, input_buffer, NN_DIR, oid + '.ann'),
      path.join(cwd, input_buffer, NN_DIR, oid + '.json') );

  exec(command, (error, stdout, stderr) => {
      // if (error) {
      //     console.log(`error: ${error.message}`);
      //     return;
      // }
      // if (stderr) {
      //     console.log(`stderr: ${stderr}`);
      //     return;
      // }
      console.log('Annotations generated for file(s) under directory: ' + fid);

      var json_path_SDP = path.join(cwd, input_buffer, SDP_DIR, oid + '.json');
      var json_SDP = JSON.parse(fs.readFileSync(json_path_SDP, 'utf8'));

      var json_path_NN = path.join(cwd, input_buffer, NN_DIR, oid + '.json');
      var json_NN = JSON.parse(fs.readFileSync(json_path_NN, 'utf8'));

      var res_json = {
        "json_SDP" : json_SDP,
        "json_NN" : json_NN,
        "input_buffer" : input_buffer,
        "oid" : oid,
      };
      res.json(res_json);
  });

});

app.post('/get_example', function(req, res) {
  var cwd = process.cwd();
  var oid = req.body.ex_num.toString();
  var dir_name = "ex" + oid;
  var input_buffer = EXAMPLE_DIR + "/" + dir_name;
  console.log(input_buffer);

  var json_path_SDP = path.join(cwd, input_buffer, SDP_DIR, oid + '.json');
  var json_SDP = JSON.parse(fs.readFileSync(json_path_SDP, 'utf8'));

  var json_path_NN = path.join(cwd, input_buffer, NN_DIR, oid + '.json');
  var json_NN = JSON.parse(fs.readFileSync(json_path_NN, 'utf8'));

  var res_json = {
    "json_SDP" : json_SDP,
    "json_NN" : json_NN,
  };
  res.json(res_json);
});

app.get('/download/', function(req, res){
  const file_path = req.query.file_path;
  console.log("Download file request @ " + file_path);
  res.download(file_path); // Set disposition and send it.
});

// viewed at http://localhost:3000
var server = app.listen(3000, "127.0.0.1", function (){
    var host = server.address().address;
    var port = server.address().port;
    console.log('KGE server listening at http://%s:%s', host, port);
});
server.timeout = 1000000;
