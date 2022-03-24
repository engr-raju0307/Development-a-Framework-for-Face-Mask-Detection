<?php

    $servername = "localhost";
    $username = "hospzmtl_ponds_user";		
    $password = "Welcome@1234";			
    $dbname = "hospzmtl_unilever_ponds_db";
// Create connection
	$conn = new mysqli($servername, $username, $password, $dbname);
	global $conn;
// Check connection
	if ($conn->connect_error) {
        die("Database Connection failed: " . $conn->connect_error);
    }
?>
