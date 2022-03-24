<?php

require('connectDB.php');



 if($_SERVER['REQUEST_METHOD']=='POST'){


 $date = $_POST['date'];
 $time = $_POST['time'];
 $imgcode = $_POST['image'];
  
     $sql = "INSERT INTO detections (date, time) VALUES (?,?)";
     $result = mysqli_stmt_init($conn);
     if (!mysqli_stmt_prepare($result, $sql)) {
         echo "SQL_Error_insert_detection";
         exit();
     }
     else{
         mysqli_stmt_bind_param($result, "ss", $date,$time);
         mysqli_stmt_execute($result);
         $imagename=mysqli_insert_id($conn);
          $path = "image/$imagename.jpg";
          file_put_contents($path,base64_decode($imgcode));
         echo "1";
         exit();
 }

 }

