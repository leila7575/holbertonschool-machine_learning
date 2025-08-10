-- Creates stored procedure for adding new score to student
DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(
	IN user_id_input INT
)
BEGIN
	UPDATE users
	SET average_score = (
		SELECT AVG(score)
		FROM corrections
		WHERE corrections.user_id = user_id_input
	)
	WHERE id = user_id_input;
END//
DELIMITER ;
