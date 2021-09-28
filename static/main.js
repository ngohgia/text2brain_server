function decodeBase64(encoded, dtype) {

    let getter = {
        "float32": "getFloat32",
        "int32": "getInt32"
    }[dtype];

    let arrayType = {
        "float32": Float32Array,
        "int32": Int32Array
    }[dtype];

    let raw = atob(encoded);
    let buffer = new ArrayBuffer(raw.length);
    let asIntArray = new Uint8Array(buffer);
    for (let i = 0; i !== raw.length; i++) {
        asIntArray[i] = raw.charCodeAt(i);
    }

    let view = new DataView(buffer);
    let decoded = new arrayType(
        raw.length / arrayType.BYTES_PER_ELEMENT);
    for (let i = 0, off = 0; i !== decoded.length;
        i++, off += arrayType.BYTES_PER_ELEMENT) {
        decoded[i] = view[getter](off, true);
    }
    return decoded;
}

function getAxisConfig() {
    let axisConfig = {
        showgrid: false,
        showline: false,
        ticks: '',
        title: '',
        showticklabels: false,
            zeroline: false,
        showspikes: false,
        spikesides: false
    };

    return axisConfig;
}

function getLighting() {
    return {};
}

function getConfig() {
    let config = {
        modeBarButtonsToRemove: ["hoverClosest3d"],
        displayLogo: false
    };

    return config;
}

function getCamera(plotDivId, viewSelectId) {
    let view = $("#" + viewSelectId).val();
    if (view === "custom") {
        try {
            return $("#" + plotDivId)[0].layout.scene.camera;
        } catch (e) {
            return {};
        }
    }
    let cameras = {
        "left": {eye: {x: -1.7, y: 0, z: 0},
                    up: {x: 0, y: 0, z: 1},
                    center: {x: 0, y: 0, z: 0}},
        "right": {eye: {x: 1.7, y: 0, z: 0},
                    up: {x: 0, y: 0, z: 1},
                    center: {x: 0, y: 0, z: 0}},
        "top": {eye: {x: 0, y: 0, z: 1.7},
                up: {x: 0, y: 1, z: 0},
                center: {x: 0, y: 0, z: 0}},
        "bottom": {eye: {x: 0, y: 0, z: -1.7},
                    up: {x: 0, y: 1, z: 0},
                    center: {x: 0, y: 0, z: 0}},
        "front": {eye: {x: 0, y: 1.7, z: 0},
                    up: {x: 0, y: 0, z: 1},
                    center: {x: 0, y: 0, z: 0}},
        "back": {eye: {x: 0, y: -1.7, z: 0},
                    up: {x: 0, y: 0, z: 1},
                    center: {x: 0, y: 0, z: 0}},
    };
    return cameras[view];

}

function getLayout(plotDivId, viewSelectId, blackBg) {

    let camera = getCamera(plotDivId, viewSelectId);
    let axisConfig = getAxisConfig();

    let container = $('#' + plotDivId);

    let width = container.width() * .9;
    let height = width * 2 / 3;

    let layout = {
        height: height, width: width,
        margin: {l:0, r:0, b:0, t:0, pad:0},
        hovermode: false,
        paper_bgcolor: blackBg ? '#000': '#fff',
        axis_bgcolor: '#333',
        scene: {
            camera: camera,
            xaxis: axisConfig,
            yaxis: axisConfig,
            zaxis: axisConfig
        }
    };

    return layout;

}

function updateLayout(plotDivId, viewSelectId, blackBg) {
    let layout = getLayout(
        plotDivId, viewSelectId, blackBg);
    Plotly.relayout(plotDivId, layout);
}

function textColor(black_bg){
    if (black_bg){
        return "white";
    }
    return "black";
}

function addColorbar(rawColorscale, cmax, colorbarId, legendId) {
    // hack to draw the colorbar
    let width = 128;
    let height = 16;
    let cv  = document.getElementById(colorbarId);
    cv.style.border = "solid 1px rgb(102, 102, 102)";
    let ctx = cv.getContext('2d');

    const grayColor = 'rgb(127, 127, 127)';
    let colorscale = [];
  
    let firstGrayColorIdx = -1;
    for (let i = parseInt(rawColorscale.length / 2); i < rawColorscale.length; i++) {
      if (rawColorscale[i][1] != grayColor) {
        colorscale.push(rawColorscale[i][1]);
        if (firstGrayColorIdx < 0)
          firstGrayColorIdx = i;
      }
    }
    colorscaleMin = cmax * (firstGrayColorIdx - rawColorscale.length / 2) / rawColorscale.length * 2;
    colorscaleMax = cmax;
    colorscaleMid = (colorscaleMin + colorscaleMax) / 2;
    let legendDiv = document.getElementById(legendId);
    legendDiv.innerHTML  = "<div class='colorbar-legend-number colorbar-legend-number--min'>" + colorscaleMin.toFixed(2) + "</div>";
    legendDiv.innerHTML += "<div class='colorbar-legend-number colorbar-legend-number--mid'>" + colorscaleMid.toFixed(2) + "</div>";
    legendDiv.innerHTML += "<div class='colorbar-legend-number colorbar-legend-number--max'>" + colorscaleMax.toFixed(2) + "</div>";
    legendDiv.style.width = width + "px";
    legendDiv.style.height = height + "px";

    step = width / colorscale.length;
    for(let i = 0; i < colorscale.length; i++) {
        ctx.beginPath();

        let color = colorscale[i];
        ctx.fillStyle = color;

        ctx.fillRect(i * step, 0, step, 150);
    }
}

function decodeHemisphere(surfaceInfo, surface, hemisphere){
    let info = surfaceInfo[surface + "_" + hemisphere];

    for (let attribute of ["x", "y", "z"]) {
        if (!(attribute in info)) {
            info[attribute] = decodeBase64(
                info["_" + attribute], "float32");
        }
    }

    for (let attribute of ["i", "j", "k"]) {
        if (!(attribute in info)) {
            info[attribute] = decodeBase64(
                info["_" + attribute], "int32");
        }
    }

}

function makePlot(surfaceMapInfo, surface, hemisphere, surfacePlotId, viewSelectId, colorbarId, colorbarLegendId) {
    decodeHemisphere(surfaceMapInfo, surface, hemisphere);
    info = surfaceMapInfo[surface + "_" + hemisphere];
    info["type"] = "mesh3d";
    info["vertexcolor"] = surfaceMapInfo["vertexcolor_" + hemisphere];

    let data = [info];

    info['lighting'] = getLighting();
    let layout = getLayout(surfacePlotId, viewSelectId,
                           surfaceMapInfo["black_bg"]);
    layout['title'] = {
        text: surfaceMapInfo['title'],
        font: {size: surfaceMapInfo["title_fontsize"],
               color: textColor(surfaceMapInfo["black_bg"])},
        yref: 'paper',
        y: .95};
    let config = getConfig();

    Plotly.react(surfacePlotId, data, layout, config);
    
    addColorbar(surfaceMapInfo["colorscale"],
                surfaceMapInfo["cmax"],
                colorbarId,
                colorbarLegendId)
}

function addPlot(surfaceMapInfo) {
    let lhKind = $("#lh-select-kind").val();
    makePlot(surfaceMapInfo, lhKind, "left", "lh-surface-plot", "lh-select-view", "lh-colorbar", "lh-colorbar-legend");
    let rhKind = $("#rh-select-kind").val();
    makePlot(surfaceMapInfo, rhKind, "right", "rh-surface-plot", "rh-select-view", "rh-colorbar", "rh-colorbar-legend");
}

function lhSurfaceRelayout(surfaceMapInfo){
    return updateLayout("lh-surface-plot", "lh-select-view", surfaceMapInfo["black_bg"]);
}

function rhSurfaceRelayout(surfaceMapInfo){
    return updateLayout("rh-surface-plot", "rh-select-view", surfaceMapInfo["black_bg"]);
}

$(document).ready(
    function() {
        $("#lh-surface-plot").mouseup(function() {
            $("#lh-select-view").val("custom");
        });
        $("#rh-surface-plot").mouseup(function() {
            $("#rh-select-view").val("custom");
        });
    }
);


(function () {
  'use strict';

  angular.module('BrainInterpreterApp', ['ngSanitize'])

  .controller('BrainInterpreterController', ['$scope', '$log', '$http', '$timeout',
    function($scope, $log, $http, $timeout) {
      $scope.submitButtonText = 'Submit';
      $scope.loading = false;
      $scope.commentLoading = false;
      $scope.error = '';
      $scope.success = false;
      $scope.commentSuccess = false;
      $scope.query = '';
      $scope.comment = '';
      $scope.relatedArticles = [];

      $scope.getResults = function() {
        var userInput = $scope.query;

        $http.post('/check', {"query": userInput}).success(function(data, status, headers, config) {
          $scope.loading = true;

          if (status === 200) {
            let resultId = data["id"];
            if (resultId > 0) {
              $http.get('/results/' + resultId).
                success(function(data, status, headers, config) {
                  if (status === 200) {
                    $scope.renderData(data, "");
                  } else
                    $scope.error = 'Error retrieving prediction';
                });
            } else {
              $http.post('/start', {"query": userInput}).then(function successCallback(response) {
                $scope.pollForResults(response.data);
                $scope.loading = true;
              }, function errorCallback(response) {
                $scope.loading = false;
              });
            }
          }
        });
      };

      $scope.createComment = function() {
        let query = $scope.query;
        let comment = $scope.comment;

        if (comment.length > 0)
          $http.post('/create-comment', {
            "query": $scope.query,
            "comment": $scope.comment,
          }).success(function(data, status, headers, config) {
            $scope.commentLoading = true;

            if (status === 200) {
              $scope.commentLoading = false;
              $scope.commentSuccess = true;
            } else {
              $scope.commentLoading = false;
	      $scope.error = 'Error creating comment';
            }
          });
      };

      $scope.feelBrainy = function() {
        $http.post('/feel-brainy', {}).success(function(data, status, headers, config) {
          $scope.loading = true;

          if (status === 200) {
            let resultId = data["id"];
            $scope.query = data["query"];
            if (resultId > 0) {
              $http.get('/results/' + resultId).
                success(function(data, status, headers, config) {
                  if (status === 200) {
                    $scope.renderData(data, "");
                  } else
                    $scope.error = 'Error retrieving prediction';
                });
            } else {
              $scope.error = "Error retrieving an example";
            }
          }
        });
      }

      $scope.pollForResults = function(jobId) {
        var timeout = "";

        var poller = function() {
          $http.get('/job_results/' + jobId).
            success(function(data, status, headers, config) {
              if (status === 202) {
                $log.log(data, status);
              } else if (status === 200){
                $scope.renderData(data, timeout);
                return false;
              }
              timeout = $timeout(poller, 2000);
            });
        };
        poller();
      };

      $scope.renderData = function(data, timeout) {
          let surfaceInfo = data['surface_info'];
          let relatedArticles = data['related_articles'];
          addPlot(surfaceInfo);
          $scope.relatedArticles = relatedArticles;
          if (timeout !== "")
            $timeout.cancel(timeout);
          $scope.loading = false;
          $scope.success = true;
      
          $("#lh-select-kind").change({ surfaceMapInfo: surfaceInfo }, function(ev) {
            addPlot(ev.data.surfaceMapInfo);
          });
          $("#rh-select-kind").change({ surfaceMapInfo: surfaceInfo }, function(ev) {
            addPlot(ev.data.surfaceMapInfo);
          });
      
          $("#lh-select-view").change({ surfaceMapInfo: surfaceInfo }, function(ev) {
            lhSurfaceRelayout(ev.data.surfaceMapInfo);
          });
          $("#rh-select-view").change({ surfaceMapInfo: surfaceInfo }, function(ev) {
            rhSurfaceRelayout(ev.data.surfaceMapInfo);
          });
      
          $(window).resize({ surfaceMapInfo: surfaceInfo }, function(ev) {
            lhSurfaceRelayout(ev.data.surfaceMapInfo);
            rhSurfaceRelayout(ev.data.surfaceMapInfo);
          });
      };

      $scope.scrollToTop = function() {
          window.scrollTo({ top: 0, behavior: 'smooth' });
      };

      $scope.scrollToHowItWorks = function() {
         let el = document.getElementById("how-it-works");
         el.scrollIntoView({behavior: "smooth"});
      }

      $scope.scrollToFeedback = function() {
         let el = document.getElementById("feedback");
         el.scrollIntoView({behavior: "smooth"});
      }
  }]);
}());
