<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge">
	<meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="">
	 <meta name="keywords" content="">
  	<meta name="author" content="Mahmudul Haque">
	<title>Mask Detection</title>
	<script src="js/jquery-1.10.2.min.js"></script>
    <script src="js/jquery-ui.js"></script>
    <script src="js/bootstrap.min.js"></script>
    <link rel="stylesheet" href="css/bootstrap.min.css">
    <link href = "css/jquery-ui.css" rel = "stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.6.3/css/font-awesome.min.css">
<link href="https://fonts.googleapis.com/css?family=Roboto+Condensed|Rubik" rel="stylesheet">
    <!-- Custom CSS -->
    <link href="css/style.css" rel="stylesheet">
    <link href="//db.onlinewebfonts.com/c/a4e256ed67403c6ad5d43937ed48a77b?family=Core+Sans+N+W01+35+Light" rel="stylesheet" type="text/css"/>
    <link rel="stylesheet" href="css/form.css" type="text/css">
    
 <style>

.export {margin: 0px 0px 10px 20px; background-color:#900C3F ;font-family:cursive;border-radius: 7px;width: 145px;height: 28px;color: #FFC300; border-color: #581845;font-size:17px}
.export:hover {cursor: pointer;background-color:#C70039}
#table {
    font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
    border-collapse: collapse;
    width: 100%;
}

#table td, #table th {
    border: 1px solid #ddd;
    padding: 8px;
     opacity: 1.6;
}
</style>
</head>
<body>
	<header>
       <div class="container">
         <div id="branding">
          <h1 style="text-align: center;"><span class="highlight">Mask Detection System</span></h1>
         </div>
        
       </div>
    </header> 

    <div class="container">
        <div class="row">
          <div id="Log"></div>
          <TABLE  id="table">
  <TR>
    <TH>ID</TH>
    <TH>Date</TH>
    <TH>Time</TH>
    <TH>Image</TH>
  </TR>
<?php 
//Connect to database
require('connectDB.php');


$sql ="SELECT * FROM detections ORDER BY id DESC";
$result=mysqli_query($conn,$sql);
if (mysqli_num_rows($result) > 0)
{
  while ($row = mysqli_fetch_assoc($result))
    {
?>
   <TR>
      <TH><?php echo $row['id']?></TH>
      <TH><?php echo $row['date']?></TH>
      <TH><?php echo $row['time']?></TH>
      <TH><a href="image/<?php echo $row['id']?>.jpg" target="_blank" title="Click">Click to view</a></TH>
   </TR>
<?php   
    }
}
?>
</TABLE>






          </div>

    </div>

  




</body>
</html>

<script>
  $(document).ready(function(){
    setInterval(function(){
      $.ajax({
        url: "load_violations.php"
        }).done(function(data) {
        $('#Log').html(data);
      });
    },3000);
  });
</script>