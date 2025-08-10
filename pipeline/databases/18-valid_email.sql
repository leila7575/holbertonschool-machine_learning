-- Creates trigger to decrease quantity of item based on orders of this item
DROP TRIGGER IF EXISTS  valid_email_reset;

DELIMITER //
CREATE TRIGGER valid_email_reset
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
	IF OLD.email <> NEW.email THEN
		SET NEW.valid_email = 0;
	END IF;
END//
DELIMITER ;
