(function () {
  'use strict';

  angular.module('GivingProfApp', [])

  .controller('GivingProfController', ['$scope', '$log', '$http', '$timeout',
    function($scope, $log, $http, $timeout) {
      $scope.submitButtonText = 'Submit';
      $scope.loading = false;
      $scope.urlerror = false;

      $scope.getResults = function() {
        // get the URL from the input
        var userInput = $scope.urlKey;

        // fire the API request
        $http.post('/start', {"url_key": userInput}).then(function successCallback(response) {
          $log.log(response);
          pollForResults(response.data);
          $scope.loading = true;
          $scope.submitButtonText = 'Loading...';
          $scope.urlerror = false;
        }, function errorCallback(response) {
          $log.log("Error");
          $log.log(response);
        });
      };

    function pollForResults(jobID) {
      var timeout = "";
    
      $log.log("Polling");
      var poller = function() {
        $http.get('/results/'+jobID).
          success(function(data, status, headers, config) {
            if(status === 202) {
              $log.log(data, status);
            } else if (status === 200){
              $log.log(data);

              const body = document.getElementById("searchResult");
              var new_body = document.createElement("tbody");
              new_body.id = "searchResult";
              data.forEach( item => {
                let row = new_body.insertRow();
                let urlCell = row.insertCell(0);
                var a = document.createElement('a');
                a.href = item[0];
                a.innerHTML = item[0];
                urlCell.appendChild(a);

                let sim_score = row.insertCell(1);
                sim_score.innerHTML = item[1];
              });
              body.parentNode.replaceChild(new_body, body);

              $scope.results = data;
              $timeout.cancel(timeout);
              return false;
            }
            timeout = $timeout(poller, 2000);
          });
      };
      poller();
    }
  }]);
}());
