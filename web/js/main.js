// global constants
const DNA_CHARS = ['A', 'C', 'G', 'T'];
const DERMOSCOPIC_FEATURES = ['asymmetry', 'border', 'colour', 'diameter', 'globules', 'blotches', 'red',
    'rosettes','regression', 'blue', 'atypical', 'streaks'];
const API_DNA = "https://melanoma-detection-tool-api.herokuapp.com/predict-melanoma/pps";
const API_DERMOSCOPIC = "https://melanoma-detection-tool-api.herokuapp.com/predict-melanoma/dermoscopic-images";

/***
 * The method handles melanoma detection by DNA from submission
 */
function onCheckResult() {
    document.getElementById("results-dna").setAttribute("style", "display: none;");
    document.getElementById("results-image").setAttribute("style", "display: none;");
    document.getElementById("input-validation-error").setAttribute("style", "display: none;");
    document.getElementById("server-error").setAttribute("style", "display: none;");
    const type = document.querySelector('input[name="selection"]:checked').value;

    if(type === "typeDermoscopicImage") {
        processByDermoscopicImage();
    } else if(type === "typeDNA") {
        processByDNA();
    }
}

/***
 * The method handles the form data and
 * call rest api for melanoma detection using DNA
 */
function processByDNA() {
    const geneName = document.querySelector('#geneName').value;
    const tier = Number(document.querySelector('#tier').value);
    const tumorOrigin = document.querySelector('#tumorOrigin').value;
    const age = Number(document.querySelector('#age').value);
    const mutatedDNASeq = document.querySelector('#mutatedDNA').value.trim();

    // change must be between 0-120 in years
    if(!(age >= 0 && age <= 120)) {
        showValidationErrorMessage("Age must be in between 0 - 120");
        return;
    }

    // DNA must contains only A,C,G,T chars
    if(!isValidDNA(mutatedDNASeq)) {
        showValidationErrorMessage("DNA Sequence can be only contained 'A', 'C', 'G', 'T' characters!");
        return;
    }

    // set button default mode
    setButtonToLoadingMode();

    // All inputs valid now!
    const request_data = {
        gene : geneName,
        tumor : tumorOrigin,
        tier : tier,
        age : age,
        mutated_dna : mutatedDNASeq
    };

    // call ajax request
    $.ajax({
        method: "POST",
        crossDomain: true,
        dataType: "json",
        crossOrigin: true,
        async: true,
        headers: {
            'Content-Type': 'application/json'
        },
        data: JSON.stringify(request_data),
        url: API_DNA,
        success: function (result) {
            showResultByDNA(result);
            setButtonToDefaultMode();
        },
        error: function (error) {
            showServerErrorMessage(error.message);
            setButtonToDefaultMode();
        }
    });
}

/***
 * The method handles the form data and
 * call rest api for melanoma detection using dermoscopic images
 */
function processByDermoscopicImage() {
    let fd = new FormData();
    let files = $('#dermoscopic-img')[0].files;
    if(files.length > 0 ) {
        fd.append('image',files[0]);
        setButtonToLoadingMode();
        $.ajax({
            method: "POST",
            crossDomain: true,
            dataType: "json",
            crossOrigin: true,
            async: true,
            contentType: false,
            processData: false,
            data: fd,
            url: API_DERMOSCOPIC,
            success: function (result) {
                showResultByDermoscopicImage(result);
                setButtonToDefaultMode();
            },
            error: function (error) {
                showServerErrorMessage(error.message);
                setButtonToDefaultMode();
            }
        });
    } else {
        showValidationErrorMessage("Please input the dermoscopic image!");
    }
}

/***
 * The method show results received from the melanoma detection by Dermoscopic Image API
 * @param result - JSON of response received from melanoma detection
 *                 by Dermoscopic Image API
 */
function showResultByDermoscopicImage(result) {
    document.getElementById("results-image").removeAttribute("style");
    DERMOSCOPIC_FEATURES.map((feature, index) => {
        if(result[2][index] === 0)
            $(`#${feature}`).text("No");
        else
            $(`#${feature}`).text("Yes");
    });
    document.getElementById("classification-2").innerText = result[0];
    document.getElementById("confidence-level-2").innerText = result[1].toFixed(2);
    if(result[0] === "Negative")
        document.getElementById("risk-level-alert-2").setAttribute("class", "alert alert-success");
    else
        document.getElementById("risk-level-alert-2").setAttribute("class", "alert alert-danger");
}

/***
 * The method show results received from the melanoma detection by DNA API
 * @param result - JSON of response received from melanoma detection
 *                 by DNA API
 */
function showResultByDNA(result) {
    const prob_no = (Number(result['probability'][0]) * 100).toFixed(2);
    const prob_yes = (Number(result['probability'][1]) * 100).toFixed(2);
    document.getElementById("results-dna").removeAttribute("style");

    if(prob_yes >= 50.0) {
        document.getElementById("risk-level-alert-1").setAttribute("class", "alert alert-danger");
        document.getElementById("confidence-level-1").innerText = prob_yes.toString();
        document.getElementById("classification-1").innerText = "Positive";
    } else {
        document.getElementById("risk-level-alert-1").setAttribute("class", "alert alert-success");
        document.getElementById("confidence-level-1").innerText = prob_no.toString();
        document.getElementById("classification-1").innerText = "Negative";
    }

    $('#pss_changes > tr > td').remove();
    let pps_tr = "<tr>";
    const pps_values = result['pps'];
    for(let i=0; i<pps_values.length; i++) {
        pps_tr += "<td>" + pps_values[i].toString() + "</td>";
    }
    pps_tr += "</tr>";
    $('#pss_changes').append(pps_tr);
}

/***
 * The method set 'Check Result' button to default mode
 */
function setButtonToDefaultMode() {
    document.getElementById("btn-loading-spinner").removeAttribute("class");
    document.getElementById("btn-check-result").removeAttribute("disabled");
}

/***
 * The method set 'Check Result' button to be disabled and loading mode
 */
function setButtonToLoadingMode() {
    document.getElementById("btn-loading-spinner")
        .setAttribute("class", "spinner-border spinner-border-sm");
    document.getElementById("btn-check-result").setAttribute("disabled", "true");
}

/***
 * The method shows input validation error messages
 * @param errorMessage - instance of String that indicates the error message
 */
function showValidationErrorMessage(errorMessage) {
    document.getElementById("input-validation-error").setAttribute("style", "display: block;");
    document.getElementById("input-validation-error-message").innerText = errorMessage;
}

/***
 * The method shows input validation error messages
 * @param errorMessage - instance of String that indicates the error message
 */
function showServerErrorMessage(errorMessage) {
    document.getElementById("server-error").setAttribute("style", "display: block;");
    document.getElementById("server-error-message").innerText = errorMessage;
}

/***
 * This function switch the melanoma diagnosis method based on user option
 */
function onSwitchForm() {
    // references
    const type = document.querySelector('input[name="selection"]:checked').value;
    // switch forms
    if(type === "typeDermoscopicImage") {
        document.getElementById("form-skin-image").removeAttribute("style");
        document.getElementById("form-dna").setAttribute("style", "display: none;");
    } else if(type === "typeDNA") {
        document.getElementById("form-dna").removeAttribute("style");
        document.getElementById("form-skin-image").setAttribute("style", "display: none;");
    }
}

/***
 * The method validates the DNA sequence
 * @param seq -  instance of String
 * @returns {boolean} returns True/False.
 *                    True - Sequence is valid.
 *                    False - Sequence is invalid.
 */
function isValidDNA(seq) {
    if(seq.length === 0) return false;

    for (let i = 0; i < seq.length; i++) {
        if(!DNA_CHARS.includes(seq.charAt(i))) {
            return false;
        }
    }
    return true;
}