-- Creates stored procedure for adding new score to student
DROP PROCEDURE IF EXISTS AddBonus;

DELIMITER //
CREATE PROCEDURE AddBonus(
	IN user_id INT,
	IN project_name VARCHAR(255),
	IN score INT
)
BEGIN

	DECLARE project_id INT;

	INSERT INTO projects (name)
	VALUES (project_name)
	ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id);

	SET project_id = LAST_INSERT_ID();

	INSERT INTO corrections (user_id, project_id, score)
	VALUES (user_id, project_id, score);

END//
DELIMITER ;
