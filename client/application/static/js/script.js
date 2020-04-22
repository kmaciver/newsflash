var textAreaElement = document.querySelector("#text-wrapper-input");

function recordText(e) {
    if (e.keyCode===32){ //checking space bar for loading words
    let textEntered = textAreaElement.value;

    var oReq = new XMLHttpRequest();
    oReq.open("POST", "/read_text", true);
    oReq.send(textEntered)
    
    oReq.onload = function () {
        if (oReq.readyState === oReq.DONE) {
            if (oReq.status === 200) {
                if (oReq.responseText!=textEntered){
                    console.log(oReq.responseText);
                    var currentPos = textAreaElement.selectionStart;
                    let renderposition = getCursorXY(textAreaElement,currentPos);
                    // console.log(renderposition)
                    var obj = JSON.parse(oReq.responseText);
                    renderDropdown(renderposition, obj,currentPos);
                }
            }
        }
    };   
    }
}

textAreaElement.addEventListener("keypress", recordText, false);

//================================================================//
// Restart Button

document.getElementById("comp-k8z7hr26link").addEventListener('click', eraseTextare);

function eraseTextare() {
  textAreaElement.value = '';
};


//================================================================//
// Save button

document.getElementById("savetextbutton").addEventListener('click', download);

function download() {
  var a = document.getElementById("savebuttonfile");
  var text = document.querySelector("#text-wrapper-input").value;
  // console.log(text)
  var file = new Blob([text], {type: 'text/plain'});
  a.href = URL.createObjectURL(file);
  a.download = 'myarticle.txt';
}

//================================================================//

// modified code from https://medium.com/@jh3y/how-to-where-s-the-caret-getting-the-xy-position-of-the-caret-a24ba372990a

// Gets the position where the dropdown should be rendered
const getCursorXY = (input, selectionPoint) => {
    const {
      offsetLeft: inputX,
      offsetTop: inputY,
    } = input
    // create a dummy element that will be a clone of our input
    const div = document.createElement('div')
    // get the computed style of the input and clone it onto the dummy element
    const copyStyle = getComputedStyle(input)
    for (const prop of copyStyle) {
      div.style[prop] = copyStyle[prop]
    }
    div.style.position='absolute'
    // we need a character that will replace whitespace when filling our dummy element if it's a single line <input/>
    const swap = '.'
    const inputValue = input.tagName === 'INPUT' ? input.value.replace(/ /g, swap) : input.value
    // set the div content to that of the textarea up until selection
    const textContent = inputValue.substr(0, selectionPoint)
    // set the text content of the dummy element div
    div.textContent = textContent
    if (input.tagName === 'TEXTAREA') div.style.height = 'auto'
    // create a marker element to obtain caret position
    const span = document.createElement('span')
    // give the span the textContent of remaining content so that the recreated dummy element is as close as possible
    span.textContent = inputValue.substr(selectionPoint) || '.'
    // append the span marker to the div
    div.appendChild(span)
    // append the dummy element to the body
    document.querySelector("#text-area").insertBefore(div, document.querySelector("#text-area").childNodes[0])
    // get the marker position, this is the caret position top and left relative to the input
    const { offsetLeft: spanX, offsetTop: spanY } = span
    // lastly, remove that dummy element
    // NOTE:: can comment this out for debugging purposes if you want to see where that span is rendered
    document.querySelector("#text-area").removeChild(div)
    // return an object with the x and y of the caret. account for input positioning so that you don't need to wrap the input
    return {
      x: inputX + spanX,
      y: inputY + spanY+25,
    }
  }

// //================================================================//
//Function to render dropdown on position with correct candidates

const renderDropdown = (renderposition, candidates, currentPos) => {
    // load the candidates in the dropdown
    var dropdown = document.getElementById("myInput")
    dropdown.style.position = 'absolute';
    dropdown.style.top = renderposition.y;
    dropdown.style.left = renderposition.x;
    
    autocomplete(dropdown, candidates, renderposition, currentPos);
}


function autocomplete(inp, candidates, renderposition, currentPos) {
    /*the autocomplete function takes two arguments,
    the text field element and an array of possible autocompleted values:*/
    var currentFocus;
    /*execute a function when someone writes in the text field:*/
    document.querySelector("#text-wrapper-input").addEventListener("input", function(e){
        var a, b, i, val = document.querySelector("#text-wrapper-input").value.substr(currentPos);
        /*close any already open lists of autocompleted values*/
        closeAllLists();
        if (!val) { return false;}
        currentFocus = -1;
        /*create a DIV element that will contain the items (values):*/
        a = document.createElement("DIV");
        a.setAttribute("id", this.id + "autocomplete-list");
        a.setAttribute("class", "autocomplete-items");
        a.style.display = 'block';
        a.style.position = 'absolute';
        a.style.top = renderposition.y;
        a.style.left = renderposition.x;
        /*append the DIV element as a child of the autocomplete container:*/
        this.parentNode.appendChild(a);
        /*for each item in the array...*/
        var candidateWords = Object.keys(candidates)
        let filterKeys =Array()
        for (i = 0; i < candidateWords.length; i++) {
          /*check if the item starts with the same letters as the text field value:*/
          if (candidateWords[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
            filterKeys.push(Object.keys(candidates)[i]);
          };
        }
        // Create a filtered object based on the filtered candidates
        Object.filter = (obj, predicate) => Object.fromEntries(Object.entries(obj).filter(predicate));
        var filtered = Object.filter(candidates, ([name, probability]) => filterKeys.includes(name));
        
        var sortable = [];
        for (var words in filtered) {
            sortable.push([words, filtered[words]]);
        }
        sortable.sort(function(a, b) {
          return b[1] - a[1];
        });

        var objSorted = {}
        sortable.forEach(function(item){
        objSorted[item[0]]=item[1]
        });
        
        let minProb = Math.min(...Object.values(objSorted))
        let maxProb = Math.max(...Object.values(objSorted))
        const lowerBound = 0.4
        const upperBound = 1
        
        Object.keys(objSorted).forEach(function(key){ 
          if (Object.keys(objSorted).length>1){
            objSorted[key] = ((upperBound-lowerBound)*((objSorted[key] - minProb)/(maxProb-minProb)))+lowerBound}
          else{
            objSorted[key] = upperBound
          }})       
        
        // console.log(objSorted)

        var arr = Object.keys(objSorted);
        for (i = 0; i < arr.length; i++) {
          if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()){
            /*create a DIV element for each matching element:*/
            b = document.createElement("DIV");
            /*make the matching letters bold:*/
            b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
            b.innerHTML += arr[i].substr(val.length);
            /*insert a input field that will hold the current array item's value:*/
            b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
            /*execute a function when someone clicks on the item value (DIV element):*/
            b.style.opacity = objSorted[arr[i]]
            b.style.backgroundColor = '#C7C5ED'
            b.addEventListener("click", function(e) {
                /*insert the value for the autocomplete text field:*/
                document.querySelector("#text-wrapper-input").value = document.querySelector("#text-wrapper-input").value.substr(0,currentPos) + this.getElementsByTagName("input")[0].value;
                $('#text-wrapper-input').focus()
                /*close the list of autocompleted values,
                (or any other open lists of autocompleted values:*/
                closeAllLists();
            });
            a.appendChild(b);
          };
        }


    });
    /*execute a function presses a key on the keyboard:*/
    document.querySelector("#text-wrapper-input").addEventListener("keydown", keyboardClicks)
    
    function keyboardClicks(e) {
        var x = document.getElementById(this.id + "autocomplete-list");
        if (x) x = x.getElementsByTagName("div");
        if (e.keyCode == 40) {
          /*If the arrow DOWN key is pressed,
          increase the currentFocus variable:*/
          currentFocus++;
          /*and and make the current item more visible:*/
          addActive(x);
        } else if (e.keyCode == 38) { //up
          /*If the arrow UP key is pressed,
          decrease the currentFocus variable:*/
          currentFocus--;
          /*and and make the current item more visible:*/
          addActive(x);
        } else if (e.keyCode == 39) {
          /*If the left arrow is pressed*/
          if (currentFocus > -1) {
            /*and simulate a click on the "active" item:*/
            if (x) x[currentFocus].click();
          }
          else {
            currentFocus = 0
            if (x) x[currentFocus].click();
          }
        }
    };
    
    function addActive(x) {
      /*a function to classify an item as "active":*/
      if (!x) return false;
      /*start by removing the "active" class on all items:*/
      removeActive(x);
      if (currentFocus >= x.length) currentFocus = 0;
      if (currentFocus < 0) currentFocus = (x.length - 1);
      /*add class "autocomplete-active":*/
      // console.log(x[currentFocus])
      x[currentFocus].classList.add("autocomplete-active");
    }
    function removeActive(x) {
      /*a function to remove the "active" class from all autocomplete items:*/
      for (var i = 0; i < x.length; i++) {
        x[i].classList.remove("autocomplete-active");
      }
    }
    function closeAllLists(elmnt) {
      /*close all autocomplete lists in the document,
      except the one passed as an argument:*/
      var x = document.getElementsByClassName("autocomplete-items");
      for (var i = 0; i < x.length; i++) {
        if (elmnt != x[i] && elmnt != inp) {
          x[i].parentNode.removeChild(x[i]);
        }
      }
    }
}
  
