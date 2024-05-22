import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { useEffect } from 'react';

function back() {

document.addEventListener('DOMContentLoaded', function() {
    var url = "http://192.168.0.13:8000/yolo_detection"
    useEffect(() => {
    var form1 = document.getElementById("form_1");

    form1.addEventListener('submit', function (e) {
        e.preventDefault();
        var fileInput = document.getElementById('input_pole');
        var file = fileInput.files[0];
        var formData = new FormData();
        formData.append('file', file);
        fetch(url,{
            mrthod: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            // Обработка ответа от сервера

    
        })
        .catch(error => {
            console.error('Ошибка при отправке файла:', error);
        });


    })
    });
})}

export default back;